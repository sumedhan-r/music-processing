"""MixUp data augmentation.

Based on: "mixup: Beyond Empirical Risk Minimization"
https://arxiv.org/abs/1710.09412

MixUp creates virtual training examples by mixing pairs of examples.
"""

from typing import Tuple
import numpy as np

from src.ml.supervised.data.transforms.base import BaseTransform


class MixUp(BaseTransform):
    """MixUp augmentation for creating interpolated training examples.

    Research use: Improves generalization and calibration.

    MixUp creates new examples as:
        x_mixed = lambda * x1 + (1 - lambda) * x2
        y_mixed = lambda * y1 + (1 - lambda) * y2

    where lambda ~ Beta(alpha, alpha)

    Args:
        alpha: Beta distribution parameter (higher = more mixing)
               alpha=1.0 is uniform mixing
               alpha=0.2-0.4 is common in practice
        apply_prob: Probability of applying MixUp (0.0 to 1.0)

    Example:
        >>> transform = MixUp(alpha=0.2)
        >>> # Need two examples for mixing
        >>> mixed_x, mixed_y = transform.mix(x1, y1, x2, y2)

    Note:
        This transform requires batch-level mixing (two examples at once).
        Typically used in the training loop, not in the dataset.
    """

    def __init__(self, alpha: float = 0.2, apply_prob: float = 0.5):
        self.alpha = alpha
        self.apply_prob = apply_prob

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Not applicable for MixUp (requires two examples).

        Use mix() method instead.

        Raises:
            NotImplementedError: MixUp requires two examples
        """
        raise NotImplementedError(
            "MixUp requires two examples. Use mix(x1, y1, x2, y2) instead."
        )

    def mix(
        self,
        x1: np.ndarray,
        y1: np.ndarray,
        x2: np.ndarray,
        y2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Mix two examples together.

        Args:
            x1: First example features
            y1: First example labels
            x2: Second example features
            y2: Second example labels

        Returns:
            Tuple of (mixed_x, mixed_y)
        """
        # Decide whether to apply MixUp
        if np.random.random() > self.apply_prob:
            # Return first example unchanged
            return x1, y1

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Mix features and labels
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_y = lam * y1 + (1 - lam) * y2

        return mixed_x, mixed_y

    def mix_batch(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Mix a batch of examples with shuffled pairs.

        Args:
            x_batch: Batch of features (batch_size, ...)
            y_batch: Batch of labels (batch_size, ...)

        Returns:
            Tuple of (mixed_batch_x, mixed_batch_y)
        """
        batch_size = x_batch.shape[0]

        # Sample lambda for entire batch
        lam = np.random.beta(self.alpha, self.alpha)

        # Random permutation for mixing pairs
        indices = np.random.permutation(batch_size)

        # Mix batches
        mixed_x = lam * x_batch + (1 - lam) * x_batch[indices]
        mixed_y = lam * y_batch + (1 - lam) * y_batch[indices]

        return mixed_x, mixed_y
