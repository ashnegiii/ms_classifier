class ExperimentConfig:
    
    RANDOM_SEED = 42

    EPISODE_SPLITS = [
        {
            "train": ["02-01-01", "03-04-17", "02-04-04", "miss-piggy", "the-cook", "rowlf-the-dog", "fozzie-bear"],
            "test":  ["03-04-03"],
            "val":   []
        },
        {
            "train": ["02-01-01", "03-04-17", "02-04-04"],
            "test":  ["03-04-03"],
            "val":   []
        },
        {
            "train": ["03-04-17", "02-04-04", "03-04-03", "miss-piggy", "the-cook", "rowlf-the-dog", "fozzie-bear"],
            "test":  ["02-01-01"],
            "val":   []
        },
        {
            "train": ["03-04-17", "02-04-04", "03-04-03"],
            "test":  ["02-01-01"],
            "val":   []
        },
        {
            "train": ["02-01-01", "02-04-04", "03-04-03", "miss-piggy", "the-cook", "rowlf-the-dog", "fozzie-bear"],
            "test":  ["03-04-17"],
            "val":   []
        },
        {
            "train": ["02-01-01", "02-04-04", "03-04-03"],
            "test":  ["03-04-17"],
            "val":   []
        },
        {
            "train": ["02-01-01", "03-04-17", "03-04-03", "miss-piggy", "the-cook", "rowlf-the-dog", "fozzie-bear"],
            "test":  ["02-04-04"],
            "val":   []
        },
        {
            "train": ["02-01-01", "03-04-17", "03-04-03"],
            "test":  ["02-04-04"],
            "val":   []
        }
    ]

    UNFREEZE_ENCODER_LAYERS = [2, 3]
    NUM_EPOCHS = [2]
    BATCH_SIZE = [32]
    LEARNING_RATE = [0.0001]
    WEIGHT_DECAY = [0.001]

    OUTPUT_THRESHOLD = [0.3]
    MAX_WEIGHT = [3]

    # Scheduler config
    SCHEDULER = ["CosineAnnealingLR"] #StepLR or CosineAnnealingLR

    # scheduler params
    STEP_SIZE = [1]
    GAMMA = [0.5]

    # the model's name: choose between 'vitb16', 'effnetb0', 'effnetb2', 'convnext_tiny'
    model_name = ["effnetb2", "convnext_tiny"]