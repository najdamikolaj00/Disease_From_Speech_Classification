from enum import Enum


class BaseModel(Enum):
    SpecNet = "SpecNet"
    SpecNetWithSE = "SpecNetWithSE"


class LastLayer(Enum):
    LSTM = "LSTM"
    Linear = "Linear"


class ModelKernel(Enum):
    Continuous = "Continuous"
    Window = "Window"


class InputChannels(Enum):
    SingleChannel = "SingleChannel"
    MultiChannel = "MultiChannel"
