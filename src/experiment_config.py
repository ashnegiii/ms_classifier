class ExperimentConfig:
    # Hyperparameters
    TRAIN_DATA_FRACTION = [1]
    TEST_DATA_FRACTION = [1]
    VAL_DATA_FRACTION = [1]

    UNFREEZE_ENCODER_LAYERS = [3]
    NUM_EPOCHS = [3]
    BATCH_SIZE = [32]
    LEARNING_RATE = [0.001]
    WEIGHT_DECAY = [0.001]

    OUTPUT_THRESHOLD = [0.4]
    MAX_WEIGHT = [3]

    model_name = ["vitb16"]