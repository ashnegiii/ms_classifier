from config.base_config import BaseConfig

class RQ3_GeneralizationConfig(BaseConfig):
    EPISODE_SPLITS = [
        {
            "train": ["02-01-01", "03-04-17", "02-04-04",
                      "miss-piggy", "the-cook", "rowlf-the-dog", "fozzie-bear"],
            "test": ["03-04-03"],
            "val": [],
        },
        {
            "train": ["02-01-01", "02-04-04", "03-04-03",
                      "miss-piggy", "the-cook", "rowlf-the-dog", "fozzie-bear"],
            "test": ["03-04-17"],
            "val": [],
        },
    ]
    TAG = ["RQ3"]
    DESCRIPTION = ["RQ3: Generalization to unseen episodes"]

    AUGMENTATION = [True]
    UNFREEZE_ENCODER_LAYERS = [2]

    OVERSAMPLING = [True]
    OVERSAMPLE_FACTOR = [10]

    MAX_BCE_WEIGHT = [3]
