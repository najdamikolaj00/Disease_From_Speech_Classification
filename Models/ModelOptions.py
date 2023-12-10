from enum import Enum


class BaseModel(Enum):
    ResNet18 = "ResNet18"
    SpecNet = "SpecNet"
    SpecNetWithSE = "SpecNetWithSE"


class TrainingOption(Enum):
    Pretrained = "Pretrained"
    TrainedFromScratch = "TrainedFromScratch"


class LastLayer(Enum):
    LSTM = "LSTM"
    Linear = "Linear"


class ModelKernel(Enum):
    Continuous = "Continuous"
    Window = "Window"


class InputChannels(Enum):
    SingleChannel = "SingleChannel"
    MultiChannel = "MultiChannel"
