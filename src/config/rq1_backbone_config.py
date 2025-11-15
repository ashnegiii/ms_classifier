from config.base_config import BaseConfig

class RQ1_BackboneConfig(BaseConfig):
    TAG = ["RQ1"]
    DESCRIPTION = ["RQ1: Backbone comparison"]

    MODEL_NAME = ["resnet50", "effnetb2", "convnext_tiny", "clip_vitb16"]
    PRETRAINED = [False, True]
    AUGMENTATION = [False]
    UNFREEZE_ENCODER_LAYERS = [2]

    OVERSAMPLING = [False]
    OVERSAMPLE_FACTOR = [1]
    MAX_BCE_WEIGHT = [1]
    SCHEDULER = ["CosineAnnealingLR"]
