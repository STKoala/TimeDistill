from chronos import Chronos2Pipeline
from model.teacher.base import BaseTeacher

class Chronos2Teacher(BaseTeacher):
    def __init__(self, model_id, device_map="auto"):
        self.pipeline = Chronos2Pipeline.from_pretrained(
            model_id,
            device_map=device_map
        )

    def forward(self, x):
        return self.pipeline(x)

    def get_model(self):
        return self.pipeline.model
