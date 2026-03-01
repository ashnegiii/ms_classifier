from config.base_config import BaseConfig

class RQ2_Combo1Config(BaseConfig):
    TAG = ["RQ2"]
    DESCRIPTION = ["Combo 1: AUG=True, UF=2, oversampling=False, pos_weight=True"]

    AUGMENTATION = [True]
    UNFREEZE_ENCODER_LAYERS = [2]

    OVERSAMPLING = [False]
    OVERSAMPLE_FACTOR = [1]

    MAX_BCE_WEIGHT = [5]
