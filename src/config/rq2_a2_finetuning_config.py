from config.base_config import BaseConfig

class RQ2_FinetuneDepthConfig(BaseConfig):
    TAG = ["RQ2"]
    DESCRIPTION = ["Ablation B: Fine-tuning depth (0,2,3)"]

    AUGMENTATION = [False]
    UNFREEZE_ENCODER_LAYERS = [0, 2, 3]

    OVERSAMPLING = [False]
    OVERSAMPLE_FACTOR = [1]

    MAX_BCE_WEIGHT = [1]
