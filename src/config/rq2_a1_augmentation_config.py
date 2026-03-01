from config.base_config import BaseConfig

class RQ2_AugmentationConfig(BaseConfig):
    TAG = ["RQ2"]
    DESCRIPTION = ["Ablation A: Augmentation"]

    AUGMENTATION = [False, True]
    UNFREEZE_ENCODER_LAYERS = [2]

    OVERSAMPLING = [False]
    OVERSAMPLE_FACTOR = [1]

    MAX_BCE_WEIGHT = [1]
