# models/student/builder.py

from .tiny_model import TinyChronosStudent

def create_student_model(teacher_pipeline, config):
    model_type = config.get("type", "tiny")

    if model_type == "tiny":
        return TinyChronosStudent(teacher_pipeline, config)
    else:
        raise ValueError(f"Unknown student type: {model_type}")
