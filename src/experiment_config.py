class ExperimentConfig:
    
    RANDOM_SEED = 42

    # Chooose splitting mode: "fraction" or "episode"
    SPLIT_MODE = "episode"

    # --- FRACTION-BASED ---
    TRAIN_SPLIT = [0.8]
    TEST_SPLIT  = [0.2]
    VAL_SPLIT   = [0.0]  # 0 means skip

    # --- EPISODE-BASED ---
    EPISODE_SPLITS = [
        {  # with cook1
            "train": [["02-01-01", "02-04-04","03-04-17", "cook1"]],
            "test":  [["03-04-03"]],
            "val":   [[]]
        },
        #{  # without cook1
        #    "train": [["02-01-01", "02-04-04", "03-04-03"]],
        #    "test":  [["03-04-17"]],
        #    "val":   [[]]
        #},
    ]

    UNFREEZE_ENCODER_LAYERS = [3]
    NUM_EPOCHS = [2]
    BATCH_SIZE = [32]
    LEARNING_RATE = [0.0001]
    WEIGHT_DECAY = [0.001]

    OUTPUT_THRESHOLD = [0.4]
    MAX_WEIGHT = [3]

    # the model's name: choose between 'vitb16', 'effnetb0', 'effnetb2'
    model_name = ["vitb16"]