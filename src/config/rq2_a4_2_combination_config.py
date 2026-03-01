from config.base_config import BaseConfig

class RQ2_Combo2Config(BaseConfig):
    TAG = ["RQ2"]
    DESCRIPTION = ["Combo 2: AUG=True, UF=2, oversampling=True, pos_weight=True"]

    AUGMENTATION = [True]
    UNFREEZE_ENCODER_LAYERS = [2]

    OVERSAMPLING = [True]
    OVERSAMPLE_FACTOR = [10]

    MAX_BCE_WEIGHT = [3]
