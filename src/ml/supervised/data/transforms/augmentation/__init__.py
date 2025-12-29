"""Data augmentation transforms for research.

Augmentation techniques to improve model generalization:
- Time masking: Mask time steps
- Frequency masking: Mask frequency bins
- MixUp: Mix two examples together
- Noise injection: Add random noise
"""

from src.ml.supervised.data.transforms.augmentation.masking import (
    TimeMask,
    FrequencyMask,
    SpecAugment,
)
from src.ml.supervised.data.transforms.augmentation.mixup import MixUp
from src.ml.supervised.data.transforms.augmentation.noise import AddNoise

__all__ = ["TimeMask", "FrequencyMask", "SpecAugment", "MixUp", "AddNoise"]
