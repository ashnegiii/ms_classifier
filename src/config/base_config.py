class BaseConfig:
    # fixed configurations
    EPISODE_SPLITS = [
        {
            "train": ["02-01-01", "03-04-17", "02-04-04",
                      "miss-piggy", "the-cook", "rowlf-the-dog", "fozzie-bear"],
            "test": ["03-04-03"],
            "val": [],
        },
        {
            "train": ["03-04-17", "02-04-04", "03-04-03",
                      "miss-piggy", "the-cook", "rowlf-the-dog", "fozzie-bear"],
            "test": ["02-01-01"],
            "val": [],
        },
        {
            "train": ["02-01-01", "02-04-04", "03-04-03",
                      "miss-piggy", "the-cook", "rowlf-the-dog", "fozzie-bear"],
            "test": ["03-04-17"],
            "val": [],
        },
        {
            "train": ["02-01-01", "03-04-17", "03-04-03",
                      "miss-piggy", "the-cook", "rowlf-the-dog", "fozzie-bear"],
            "test": ["02-04-04"],
            "val": [],
        },
    ]

    RANDOM_SEED = [42, 7, 2021]

    MODEL_NAME = ["clip_vitb16"]
    PRETRAINED = [True]

    NUM_EPOCHS = [15]
    PATIENCE = [3]

    BATCH_SIZE = [32]
    LEARNING_RATE = [0.0001]
    WEIGHT_DECAY = [0.001]
    OUTPUT_THRESHOLD = [0.5]

    MAX_BCE_WEIGHT = [1]

    SCHEDULER = ["CosineAnnealingLR"]

    AUGMENTATION_DESCRIPTION = [
        "RandomResizedCrop(224, scale=(0.7, 1.0)), "
        "HorizontalFlip, ColorJitter(0.3), Rotation(15°), "
        "Affine(translate=0.1), Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])"
    ]

    TAG = ["RQ1"]
    DESCRIPTION = ["BaseConfig"]
    AUGMENTATION = [False]
    UNFREEZE_ENCODER_LAYERS = [2]

    OVERSAMPLING = [False]
    OVERSAMPLE_FACTOR = [1]