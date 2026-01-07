from .base import BaseTeacher
from timesfm import TimesFM

class TimesFMTeacher(BaseTeacher):
    def __init__(self, model_id):
        self.model = TimesFM.from_pretrained(model_id)

    def forward(self, x):
        return self.model.predict(x)

    def get_model(self):
        return self.model
