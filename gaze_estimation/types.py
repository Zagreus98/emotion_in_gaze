import enum


class GazeEstimationMethod(enum.Enum):
    MPIIFaceGaze = enum.auto()
    ETHXGaze = enum.auto()


class LossType(enum.Enum):
    L1 = enum.auto()
    L2 = enum.auto()
    SmoothL1 = enum.auto()
