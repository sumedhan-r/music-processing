"""Pre-built model architectures for audio classification.

Provides ready-to-use architectures that are registered in ModelRegistry:
- SimpleCNN: Baseline CNN for audio
- ResNetAudio: ResNet-style architecture for spectrograms
- AttentionCNN: CNN with attention mechanisms
"""

from src.ml.supervised.models.pytorch.architectures.simple_cnn import SimpleCNN
from src.ml.supervised.models.pytorch.architectures.attention_cnn import AttentionCNN

__all__ = ["SimpleCNN", "AttentionCNN"]
