class Config:
    ################  COMMON  ################
    IMAGE_HEIGHT = 84
    IMAGE_WIDTH = 84
    NUM_FRAME = 4
    DISCOUNT_FACTOR = 0.99
    LOG_EPSILON = 1e-10

    ################  OPTIMIZER  ################
    ADAM_LEARNING_RATE = 1e-3
    RMS_LEARNING_RATE = 1e-3
    RMSPROP_DECAY = 0.99
    RMSPROP_MOMENTUM = 0.0
    RMSPROP_EPSILON = 0.1

    ################  Value Based  ################
    BATCH_SIZE = 32
    TRAIN_START = 50000
    TRAIN_END = 5000000
    TARGET_UPDATE_RATE = 10000
    MEMORY_SIZE = 500000
    EPSILON_START = 1.
    EPSILON_END = 0.01
    EPSILON_EXPLORATION = 1000000

    DRQN_HSIZE = 256
    DRQN_BATCH_SIZE = 8
    UNROLLING_TIME_STEPS = 4

    ################  Policy Based  ################
    ENTROPY_BETA = 0.01
    GRAD_CLIP_NORM = 40.0
    T_MAX = 5