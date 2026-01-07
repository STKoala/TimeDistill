from .base import BaseTeacher
from timesmoe import TimesMoEModel

class TimesMoETeacher(BaseTeacher):
    def __init__(self, model_id):
        self.model = TimesMoEModel.from_pretrained(model_id)

    def forward(self, x):
        return self.model(x)

    def get_model(self):
        return self.model
