class TinyChronosStudent(nn.Module):
    def __init__(self, teacher_pipeline, config):
        super().__init__()
        teacher_model = teacher_pipeline.model

        # 从 teacher 继承 embedding / tokenizer 等
        self.embedding = teacher_model.embedding

        self.num_layers = config["num_layers"]
        self.d_model = config["d_model"]
        self.d_ff = config["d_ff"]
        self.num_heads = config["num_heads"]

        self.build_layers()

    def build_layers(self):
        ...
