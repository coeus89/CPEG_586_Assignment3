from enum import Enum
class TrainingType(Enum):
    Stochastic = 0
    MiniBatch = 1
    Batch = 2