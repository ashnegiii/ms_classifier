class ExperimentConfig:

    EPISODE_SPLITS = [
        {
            "train": ["02-01-01", "03-04-17", "02-04-04", "miss-piggy", "the-cook", "rowlf-the-dog", "fozzie-bear"],
            "test":  ["03-04-03"],
            "val":   []
        }]
    """
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
        """

    RANDOM_SEED = [42, 7, 2021]

    # RQ1: How does the backbone architecture affect performance?
    TAG = ["RQ1"]
    MODEL_NAME = ["resnet50", "effnetb2", "convnext_tiny", "clip_vitb16"]
    UNFREEZE_ENCODER_LAYERS = [2]
    NUM_EPOCHS = [10]
    BATCH_SIZE = [32]
    LEARNING_RATE = [0.0001]
    WEIGHT_DECAY = [0.001]
    OUTPUT_THRESHOLD = [0.5]
    MAX_WEIGHT = [3]
    SCHEDULER = ["CosineAnnealingLR"]  # StepLR or CosineAnnealingLR

    # scheduler params
    STEP_SIZE = [1]
    GAMMA = [0.5]
