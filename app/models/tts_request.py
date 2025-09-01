from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


class ModelType(str, Enum):
    VIBEVOICE = "vibevoice"
    ZONOS = "zonos"


class TTSRequest(BaseModel):
    text: str
    model_type: Optional[ModelType] = ModelType.VIBEVOICE
    speaker_names: Optional[List[str]] = None
    cfg_scale: Optional[float] = 3.0