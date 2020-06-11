from enum import Enum
class TrainingType(Enum):
    Stochastic = 0
    MiniBatch = 1
    #Batch = 2

class ActivationType(object):
    SIGMOID = 1
    TANH = 2
    RELU = 3
    SOFTMAX = 4

class LROptimizerType(object):
    NONE = 1
    ADAM = 2

class BatchNormMode(object):
    TRAIN = 1
    TEST = 2
