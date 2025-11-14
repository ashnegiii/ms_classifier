class ExperimentConfig:

    EPISODE_SPLITS = [
        {
            "train": [
                "02-01-01",
                "03-04-17",
                "02-04-04",
                "miss-piggy",
                "the-cook",
                "rowlf-the-dog",
                "fozzie-bear",
            ],
            "test": ["03-04-03"],
            "val": [],
        },
        {
            "train": [
                "03-04-17",
                "02-04-04",
                "03-04-03",
                "miss-piggy",
                "the-cook",
                "rowlf-the-dog",
                "fozzie-bear",
            ],
            "test": ["02-01-01"],
            "val": [],
        },
        {
            "train": [
                "02-01-01",
                "02-04-04",
                "03-04-03",
                "miss-piggy",
                "the-cook",
                "rowlf-the-dog",
                "fozzie-bear",
            ],
            "test": ["03-04-17"],
            "val": [],
        },
        {
            "train": [
                "02-01-01",
                "03-04-17",
                "03-04-03",
                "miss-piggy",
                "the-cook",
                "rowlf-the-dog",
                "fozzie-bear",
            ],
            "test": ["02-04-04"],
            "val": [],
        },
    ]
    """
        {
            "train": ["02-01-01", "03-04-17", "02-04-04"],
            "test":  ["03-04-03"],
            "val":   []
        },
        
        {
            "train": ["03-04-17", "02-04-04", "03-04-03"],
            "test":  ["02-01-01"],
            "val":   []
        },
        {
            "train": ["02-01-01", "02-04-04", "03-04-03"],
            "test":  ["03-04-17"],
            "val":   []
        },
        {
            "train": ["02-01-01", "03-04-17", "03-04-03"],
            "test":  ["02-04-04"],
            "val":   []
        }
        """

    RANDOM_SEED = [42, 7, 2021]

    # ---------- RQ1: How does the backbone architecture affect performance? ----------
    # TAG = ["RQ1"]
    # MODEL_NAME = ["resnet50", "effnetb2", "convnext_tiny", "clip_vitb16"]
    # PRETRAINED = [True]
    # AUGMENTATION = [False]

    # UNFREEZE_ENCODER_LAYERS = [2]
    # NUM_EPOCHS = [15]

    # cancel training when no improvement in n epochs
    # PATIENCE = [3]

    # BATCH_SIZE = [32]
    # LEARNING_RATE = [0.0001]
    # WEIGHT_DECAY = [0.001]
    # OUTPUT_THRESHOLD = [0.5]
    # MAX_WEIGHT = [1]

    # StepLR or CosineAnnealingLR
    # SCHEDULER = ["None"]

    # ---------- RQ2: Which combination of optimization strategies most effectively improves precision and recall across all classes? ----------
    TAG = ["RQ2"]
    DESCRIPTION = ["Baseline performance"]
    MODEL_NAME = ["clip_vitb16"]
    PRETRAINED = [True]
    AUGMENTATION = [False]
    # for logging the augmentations used (if AUGMENTATION=False, description will be "No augmentation applied.")
    AUGMENTATION_DESCRIPTION = [
        "RandomResizedCrop(0.7–1.0), HorizontalFlip, ColorJitter(0.3), Rotation(15°), Affine(translate=0.1), Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])"
    ]

    OVERSAMPLING = [False]
    OVERSAMPLE_FACTOR = [1]

    UNFREEZE_ENCODER_LAYERS = [0, 2, 3]
    NUM_EPOCHS = [15]

    # cancel training when no improvement in n epochs
    PATIENCE = [3]

    BATCH_SIZE = [32]
    LEARNING_RATE = [0.0001]
    WEIGHT_DECAY = [0.001]
    OUTPUT_THRESHOLD = [0.5]
    MAX_BCE_WEIGHT = [1]

    # StepLR or CosineAnnealingLR
    SCHEDULER = ["CosineAnnealingLR"]

    # ---------- RQ3: How well does the fine-tuned model generalize to completely unseen episodes? ----------
