# time_moe_router_demo.py
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# Optional: visualization at the end
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except:
    HAS_MPL = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Synthetic dataset
# -------------------------
class SyntheticRegimeDataset(Dataset):
    """
    Produce sequences with different regimes:
      - 'stable': sinusoidal + small noise
      - 'trend' : linear trend + noise
      - 'spike' : mostly stable but with occasional spikes
    Each sample: input sequence (context_len) -> predict next target_len values
    """
    def __init__(self, n_samples=5000, context_len=64, target_len=8, regime=None):
        self.n_samples = n_samples
        self.context_len = context_len
        self.target_len = target_len
        self.regime = regime  # if None, mix regimes
        self._make_signals()

    def _make_signals(self):
        self.X = []
        self.Y = []
        self.regimes = []
        for i in range(self.n_samples):
            if self.regime is None:
                r = random.choice(['stable', 'trend', 'spike'])
            else:
                r = self.regime
            self.regimes.append(r)
            base_t = np.arange(0, self.context_len + self.target_len)
            if r == 'stable':
                freq = random.uniform(0.05, 0.2)
                phase = random.uniform(0, 2*math.pi)
                amp = random.uniform(0.5, 2.0)
                series = amp * np.sin(2*math.pi*freq*base_t + phase)
                series += np.random.normal(scale=0.1, size=base_t.shape)
            elif r == 'trend':
                slope = random.uniform(-0.05, 0.05)
                intercept = random.uniform(-1, 1)
                series = intercept + slope * base_t + 0.2 * np.sin(0.2*base_t)
                series += np.random.normal(scale=0.15, size=base_t.shape)
            elif r == 'spike':
                series = 0.5 * np.sin(0.1*base_t) + np.random.normal(scale=0.05, size=base_t.shape)
                # occasional spike in target region or context
                if random.random() < 0.3:
                    spike_pos = random.randint(self.context_len//2, self.context_len + self.target_len - 1)
                    series[spike_pos] += random.uniform(3.0, 6.0) * random.choice([1, -1])
            else:
                raise ValueError("Unknown regime")
            x = series[:self.context_len]
            y = series[self.context_len:self.context_len + self.target_len]
            self.X.append(x.astype(np.float32))
            self.Y.append(y.astype(np.float32))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.X[idx]).unsqueeze(-1),  # [context_len, 1]
            'y': torch.tensor(self.Y[idx]).unsqueeze(-1),  # [target_len, 1]
            'regime': self.regimes[idx]
        }

# -------------------------
# Simple LSTM-based expert
# -------------------------
class LSTMExpert(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=64, n_layers=1, out_len=8):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_len)  # predict target_len values in one shot
        )
        self.out_len = out_len

    def forward(self, x):
        # x: [B, T, in_dim]
        out, _ = self.lstm(x)  # out: [B, T, hidden]
        last = out[:, -1, :]   # last time-step
        pred = self.head(last) # [B, out_len]
        return pred.unsqueeze(-1)  # [B, out_len, 1]

# -------------------------j
# Router model (MLP)
# -------------------------
class Router(nn.Module):
    def __init__(self, input_len, hidden=128, n_experts=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_len, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, n_experts)
        )

    def forward(self, x):
        # x: [B, input_len]  (we'll flatten last few context values)
        logits = self.net(x)
        return logits  # raw logits; will apply softmax or gumbel-softmax outside

# -------------------------
# Gumbel-softmax helper
# -------------------------
def gumbel_softmax_sample(logits, temperature=0.5, eps=1e-20):
    # logits: [B, n_experts]
    U = torch.rand_like(logits)
    g = -torch.log(-torch.log(U + eps) + eps)
    y = logits + g
    return torch.softmax(y / temperature, dim=-1)

# -------------------------
# Training utilities
# -------------------------
def train_expert(model, dataloader, n_epochs=5, lr=1e-3):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        count = 0
        for batch in dataloader:
            x = batch['x'].to(device)  # [B, T, 1]
            y = batch['y'].to(device)  # [B, target_len, 1]
            opt.zero_grad()
            pred = model(x)  # [B, out_len, 1]
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            count += x.size(0)
        print(f"  Expert train epoch {epoch+1}/{n_epochs} loss={total_loss/count:.6f}")

def evaluate_model_on_loader(model, dl):
    model.eval()
    loss_fn = nn.MSELoss()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in dl:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            pred = model(x)
            total += loss_fn(pred, y).item() * x.size(0)
            count += x.size(0)
    return total / count

# -------------------------
# Router training (freeze experts)
# -------------------------
def train_router(router, experts, train_loader, val_loader=None,
                routing_mode='soft',  # 'soft' or 'gumbel'
                n_epochs=20, lr=1e-3, temperature=0.5, aux_load_balance=False, load_balance_coef=1e-2):
    """
    router: Router model
    experts: list of expert models (already moved to device and frozen)
    train_loader: returns batches with 'x' and 'y'
    routing_mode: 'soft' or 'gumbel' (hard-ish)
    """
    router.to(device)
    router.train()
    opt = optim.Adam(router.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n_experts = len(experts)

    for epoch in range(n_epochs):
        total_loss = 0.0
        total_aux = 0.0
        count = 0
        for batch in train_loader:
            x = batch['x'].to(device)  # [B, T, 1]
            y = batch['y'].to(device)  # [B, out_len, 1]
            B, T, _ = x.shape
            # prepare router input: flatten last k values (or use summary features)
            router_input = x[:, -32:, 0] if T >= 32 else x[:, :, 0]  # [B, k]
            logits = router(router_input)  # [B, n_experts]

            if routing_mode == 'soft':
                probs = torch.softmax(logits, dim=-1)  # [B, n_experts]
            elif routing_mode == 'gumbel':
                probs = gumbel_softmax_sample(logits, temperature=temperature)
            else:
                raise ValueError("Unknown routing_mode")

            # Collect expert predictions (no grads for experts)
            # Each expert returns [B, out_len, 1]
            expert_preds = []
            for e in experts:
                with torch.no_grad():
                    p = e(x)  # [B, out_len, 1]
                expert_preds.append(p)
            # stack: [n_experts, B, out_len, 1]
            stacked = torch.stack(expert_preds, dim=0)
            # weighted combination:
            # expand probs to [n_experts, B, out_len, 1]
            probs_exp = probs.transpose(0,1).unsqueeze(-1).unsqueeze(-1)  # [n_experts, B, 1, 1]
            combined = (probs_exp * stacked).sum(dim=0)  # [B, out_len, 1]

            loss = loss_fn(combined, y)

            # optional load balance auxiliary: encourage router to use experts evenly
            aux_loss = 0.0
            if aux_load_balance:
                # encourage mean(probs) to be uniform
                mean_probs = probs.mean(dim=0)  # [n_experts]
                target = torch.full_like(mean_probs, 1.0 / n_experts)
                aux_loss = ((mean_probs - target)**2).sum()
                loss = loss + load_balance_coef * aux_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * B
            total_aux += aux_loss if isinstance(aux_loss, float) else aux_loss.item() * B
            count += B

        avg_loss = total_loss / count
        print(f"Router epoch {epoch+1}/{n_epochs} train_loss={avg_loss:.6f}")
        if val_loader is not None:
            router.eval()
            val_loss = _eval_router(router, experts, val_loader, routing_mode, temperature)
            print(f"  Val_loss={val_loss:.6f}")
            router.train()

def _eval_router(router, experts, loader, routing_mode, temperature):
    router.eval()
    loss_fn = nn.MSELoss()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            B, T, _ = x.shape
            router_input = x[:, -32:, 0] if T >=32 else x[:,:,0]
            logits = router(router_input)
            if routing_mode == 'soft':
                probs = torch.softmax(logits, dim=-1)
            else:
                probs = gumbel_softmax_sample(logits, temperature=temperature)
            expert_preds = []
            for e in experts:
                p = e(x)
                expert_preds.append(p)
            stacked = torch.stack(expert_preds, dim=0)
            probs_exp = probs.transpose(0,1).unsqueeze(-1).unsqueeze(-1)
            combined = (probs_exp * stacked).sum(dim=0)
            loss = loss_fn(combined, y)
            total += loss.item() * B
            count += B
    return total / count

# -------------------------
# Main demo pipeline
# -------------------------
def main():
    context_len = 64
    target_len = 8

    # Create datasets: experts will be trained on regime-specific sets
    n_per_regime = 1200
    ds_stable = SyntheticRegimeDataset(n_samples=n_per_regime, context_len=context_len, target_len=target_len, regime='stable')
    ds_trend  = SyntheticRegimeDataset(n_samples=n_per_regime, context_len=context_len, target_len=target_len, regime='trend')
    ds_spike  = SyntheticRegimeDataset(n_samples=n_per_regime, context_len=context_len, target_len=target_len, regime='spike')

    # general mixed dataset for training router (hold-out validation too)
    ds_mixed = SyntheticRegimeDataset(n_samples=2000, context_len=context_len, target_len=target_len, regime=None)
    n_train = int(len(ds_mixed) * 0.8)
    train_mixed, val_mixed = torch.utils.data.random_split(ds_mixed, [n_train, len(ds_mixed)-n_train])

    dl_stable = DataLoader(ds_stable, batch_size=64, shuffle=True)
    dl_trend  = DataLoader(ds_trend,  batch_size=64, shuffle=True)
    dl_spike  = DataLoader(ds_spike,  batch_size=64, shuffle=True)
    dl_router_train = DataLoader(train_mixed, batch_size=64, shuffle=True)
    dl_router_val   = DataLoader(val_mixed, batch_size=128, shuffle=False)

    # Create experts and pretrain each on one regime (simulate pretrained expert)
    experts = []
    for i, dl in enumerate([dl_stable, dl_trend, dl_spike]):
        print(f"\n== Training expert {i} on regime {['stable','trend','spike'][i]} ==")
        model = LSTMExpert(in_dim=1, hidden_dim=64, n_layers=1, out_len=target_len).to(device)
        train_expert(model, dl, n_epochs=6, lr=1e-3)
        val_loss = evaluate_model_on_loader(model, dl)
        print(f"  Expert {i} val_loss on its regime = {val_loss:.6f}")
        experts.append(model)

    # Freeze experts
    for e in experts:
        e.eval()
        for p in e.parameters():
            p.requires_grad = False

    # Router
    router = Router(input_len=32, hidden=128, n_experts=len(experts))

    # Train router (soft)
    print("\n== Training Router: SOFT routing ==")
    train_router(router, experts, dl_router_train, val_loader=dl_router_val,
                routing_mode='soft',
                n_epochs=15, lr=5e-4, temperature=0.5, aux_load_balance=False)

    # Evaluate router
    val_loss_soft = _eval_router(router, experts, dl_router_val, routing_mode='soft', temperature=0.5)
    print("Router (soft) val loss:", val_loss_soft)

    # Optionally train a router with Gumbel (hard-ish)
    router_gumbel = Router(input_len=32, hidden=128, n_experts=len(experts))
    print("\n== Training Router: GUMBEL (hard-ish) routing ==")
    train_router(router_gumbel, experts, dl_router_train, val_loader=dl_router_val,
                routing_mode='gumbel',
                n_epochs=15, lr=5e-4, temperature=0.7, aux_load_balance=False)
    val_loss_gumbel = _eval_router(router_gumbel, experts, dl_router_val, routing_mode='gumbel', temperature=0.7)
    print("Router (gumbel) val loss:", val_loss_gumbel)

    # Quick demo inference on few samples, show chosen experts
    demo_loader = DataLoader(SyntheticRegimeDataset(n_samples=12, context_len=context_len, target_len=target_len, regime=None), batch_size=4)
    demo_batch = next(iter(demo_loader))
    x_demo = demo_batch['x'].to(device)
    y_demo = demo_batch['y'].to(device)
    router.eval()
    router_gumbel.eval()
    with torch.no_grad():
        inp = x_demo[:, -32:, 0]
        logits = router(inp)
        probs = torch.softmax(logits, dim=-1)
        topk_idx = torch.argmax(probs, dim=-1)
        print("\nSample regimes:", demo_batch['regime'])
        print("Router (soft) probs:\n", probs.cpu().numpy())
        print("Router (soft) chosen expert (argmax):", topk_idx.cpu().numpy())

        # gumbel
        logits_g = router_gumbel(inp)
        probs_g = gumbel_softmax_sample(logits_g, temperature=0.7)
        print("Router (gumbel) probs:\n", probs_g.cpu().numpy())
        print("Router (gumbel) chosen expert (argmax):", torch.argmax(probs_g, dim=-1).cpu().numpy())

    # optional plot of predictions if matplotlib available
    if HAS_MPL:
        # pick first sample and show ground truth vs soft combined prediction
        sample_x = x_demo[0:1]  # [1, T, 1]
        sample_y = y_demo[0:1]
        router.eval()
        with torch.no_grad():
            inp = sample_x[:, -32:, 0]
            probs = torch.softmax(router(inp), dim=-1)
            stacked = torch.stack([e(sample_x) for e in experts], dim=0)
            probs_exp = probs.transpose(0,1).unsqueeze(-1).unsqueeze(-1)
            combined = (probs_exp * stacked).sum(dim=0)  # [1, out_len, 1]
            pred = combined.squeeze().cpu().numpy()
            truth = sample_y.squeeze().cpu().numpy()
            ctx = sample_x.squeeze().cpu().numpy()

        plt.figure(figsize=(8,4))
        plt.plot(np.arange(len(ctx)), ctx, label='context')
        t0 = len(ctx)
        plt.plot(np.arange(t0, t0+len(truth)), truth, 'o-', label='truth')
        plt.plot(np.arange(t0, t0+len(pred)), pred, 'x--', label='router_pred')
        plt.legend()
        plt.title("Example prediction (soft router)")
        plt.savefig("my_plot.png")

if __name__ == "__main__":
    main()
