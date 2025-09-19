class ExperimentConfig:
    
    RANDOM_SEED = 42

    # Chooose splitting mode: "fraction" or "episode"
    SPLIT_MODE = "episode"

    # --- FRACTION-BASED ---
    TRAIN_SPLIT = [0.8]
    TEST_SPLIT  = [0.2]
    VAL_SPLIT   = [0.0]  # 0 means skip


    EPISODE_SPLITS = [
    {
        "train": [["02-04-04", "03-04-17", "03-04-03", "cook-1", "miss-piggy-1"]],
        "test":  [["02-01-01"]],
        "val":   [[]]
    },
    {
        "train": [["02-01-01", "03-04-17", "03-04-03", "cook-1", "miss-piggy-1"]],
        "test":  [["02-04-04"]],
        "val":   [[]]
    },
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
]

    UNFREEZE_ENCODER_LAYERS = [1, 2, 5]
    NUM_EPOCHS = [3]
    BATCH_SIZE = [32]
    LEARNING_RATE = [0.0001]
    WEIGHT_DECAY = [0.001]

    OUTPUT_THRESHOLD = [0.4]
    MAX_WEIGHT = [3]

    # the model's name: choose between 'vitb16', 'effnetb0', 'effnetb2'
    model_name = ["effnetb2", "vitb16"]