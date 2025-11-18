from config.base_config import BaseConfig

class RQ2_PosWeightConfig(BaseConfig):
    TAG = ["RQ2"]
    DESCRIPTION = ["Ablation C2: BCE pos_weight ON/OFF"]

    AUGMENTATION = [False]
    UNFREEZE_ENCODER_LAYERS = [2]

    OVERSAMPLING = [False]
    OVERSAMPLE_FACTOR = [1]

    MAX_BCE_WEIGHT = [1, 5]
