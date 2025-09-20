class ExperimentConfig:
    
    RANDOM_SEED = 42

    EPISODE_SPLITS = [
    {
        "train": [["02-01-01", "03-04-17", "02-04-04", "cook-1", "miss-piggy-1", "fozzie-bear"]],
        "test":  [["03-04-03"]],
        "val":   [[]]
    },
    {
        "train": [["02-01-01", "03-04-17", "02-04-04", "cook-1", "miss-piggy-1"]],
        "test":  [["03-04-03"]],
        "val":   [[]]
    },
    {
        "train": [["02-01-01", "03-04-03", "02-04-04", "cook-1", "miss-piggy-1", "fozzie-bear"]],
        "test":  [["03-04-17"]],
        "val":   [[]]
    },
    {
        "train": [["02-01-01", "03-04-03", "02-04-04", "cook-1", "miss-piggy-1"]],
        "test":  [["03-04-17"]],
        "val":   [[]]
    }
]

    """
    {
        "train": [["02-01-01", "02-04-04", "03-04-03", "cook-1", "miss-piggy-1"]],
        "test":  [["03-04-17"]],
        "val":   [[]]
    },
    {
        "train": [["02-01-01", "02-04-04", "03-04-17", "cook-1", "miss-piggy-1"]],
        "test":  [["03-04-03"]],
        "val":   [[]]
    },
    """
    UNFREEZE_ENCODER_LAYERS = [3]
    NUM_EPOCHS = [1]
    BATCH_SIZE = [32]
    LEARNING_RATE = [0.0001]
    WEIGHT_DECAY = [0.001]

    OUTPUT_THRESHOLD = [0.4]
    MAX_WEIGHT = [3]

    # Scheduler config
    SCHEDULER = ["StepLR"] #StepLR or CosineAnnealingLR

    # scheduler params
    STEP_SIZE = [2]
    GAMMA = [0.5]

    # the model's name: choose between 'vitb16', 'effnetb0', 'effnetb2'
    model_name = ["effnetb2"]