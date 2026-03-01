from config.base_config import BaseConfig

class RQ2_OversamplingConfig(BaseConfig):
    TAG = ["RQ2"]
    DESCRIPTION = ["Ablation C1: Oversampling ON/OFF"]

    AUGMENTATION = [False]
    UNFREEZE_ENCODER_LAYERS = [2]

    OVERSAMPLING = [True]
    OVERSAMPLE_FACTOR = [5]

    MAX_BCE_WEIGHT = [1]
