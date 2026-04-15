"""Architecture families — importing this module triggers registration."""

from model_zoo.families import cnn1d, cnn2d, deepsets, resnet, vit  # noqa: F401
from model_zoo.families.base import ArchHyperparams, ModelFamily, TrainingHyperparams

__all__ = ["ArchHyperparams", "ModelFamily", "TrainingHyperparams"]
