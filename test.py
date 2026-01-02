
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.gkd import GKDConfig, GKDTrainer
import chronos

# Initialize tokenizer and models
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
student_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
teacher_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

# Create sample datasets
train_dataset = Dataset.from_dict({
    "messages": [
        [
            {"role": "user", "content": "Hi, how are you?"},
            {"role": "assistant", "content": "I'm great thanks"},
        ]
    ] * 100
})

eval_dataset = Dataset.from_dict({
    "messages": [
        [
            {"role": "user", "content": "What colour is the sky?"},
            {"role": "assistant", "content": "The sky is blue"},
        ]
    ] * 100
})

# Configure training arguments
training_args = GKDConfig(
    output_dir="gkd-model",
    per_device_train_batch_size=1,
    lmbda=0.5,        # Controls student-generated vs teacher-generated data
    beta=0.5,         # Controls Jensen-Shannon divergence interpolation
    temperature=0.9,    # Temperature for sampling
    max_new_tokens=128  # Maximum tokens to generate
)

# Create trainer and train
trainer = GKDTrainer(
    model=student_model,
    teacher_model=teacher_model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()