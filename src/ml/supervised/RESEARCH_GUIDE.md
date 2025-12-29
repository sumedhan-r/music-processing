# Research Guide: Transforms & Models

**Design Philosophy**: Maximum flexibility for experimentation while maintaining clean code organization.

---

## 📊 Data Transforms & Preprocessing

### Quick Start: Transform Pipeline

```python
from src.ml.supervised.data.transforms import TransformPipeline
from src.ml.supervised.data.transforms.audio import AudioToMelSpectrogram
from src.ml.supervised.data.transforms.augmentation import SpecAugment
from src.ml.supervised.data.transforms.normalization import StandardNormalization

# Experiment 1: Basic pipeline
pipeline = TransformPipeline([
    AudioToMelSpectrogram(sample_rate=22050, n_mels=128),
    StandardNormalization(),
])

processed = pipeline(audio_waveform)
```

### Experiment with Different Preprocessing

```python
# Experiment 2: With augmentation
pipeline = TransformPipeline([
    AudioToMelSpectrogram(n_mels=128, fmax=8000),
    SpecAugment(time_mask_max=10, freq_mask_max=8),
    DBScaling(top_db=80),
])

# Experiment 3: MFCC instead of mel-spectrogram
from src.ml.supervised.data.transforms.audio import AudioToMFCC

pipeline = TransformPipeline([
    AudioToMFCC(n_mfcc=13, n_mels=40),
    RobustNormalization(),  # Better for outliers
])

# Experiment 4: Custom preprocessing
class MyCustomTransform(BaseTransform):
    def apply(self, data):
        # Your research idea here
        return my_processed_data

pipeline.add(MyCustomTransform())
```

### Available Transform Categories

#### 🎵 **Audio Transforms** (`transforms/audio/`)
- `AudioToSpectrogram` - STFT spectrogram
- `AudioToMelSpectrogram` - Perceptually-motivated mel-scale
- `AudioToMFCC` - Mel-frequency cepstral coefficients
- `AudioToMFCCDelta` - MFCC with velocity/acceleration

#### 🔄 **Augmentation** (`transforms/augmentation/`)
- `TimeMask` - Mask time steps (SpecAugment)
- `FrequencyMask` - Mask frequency bins
- `SpecAugment` - Combined time + freq masking
- `MixUp` - Mix pairs of examples
- `AddNoise` - Gaussian/uniform/salt-pepper noise
- `AddBackgroundNoise` - Real noise samples

#### 📏 **Normalization** (`transforms/normalization/`)
- `StandardNormalization` - Z-score (mean=0, std=1)
- `MinMaxNormalization` - Scale to [0,1] or [-1,1]
- `RobustNormalization` - Median/IQR (robust to outliers)
- `LogNormalization` - Log-scale compression
- `DBScaling` - Decibel scale
- `PerChannelNormalization` - Per-frequency normalization

---

## 🧠 Model Architectures

### Approach 1: Use Pre-built Models

```python
from src.ml.supervised.models.pytorch.registry import ModelRegistry

# List available models
print(ModelRegistry.list_models())
# Output: ['simple_cnn', 'deep_cnn', 'attention_cnn', ...]

# Create model by name
model = ModelRegistry.create("simple_cnn", num_classes=10)

# Or with custom hyperparameters
model = ModelRegistry.create(
    "attention_cnn",
    num_classes=50,
    hidden_dims=[64, 128, 256, 512],
    dropout_rate=0.5,
)
```

### Approach 2: Build from Configuration

```python
from src.ml.supervised.models.pytorch.builder import ModelBuilder

# Define architecture via config
config = {
    "blocks": [
        {"type": "conv", "in_channels": 1, "out_channels": 64, "kernel_size": 3},
        {"type": "conv", "in_channels": 64, "out_channels": 128, "kernel_size": 3},
        {"type": "channel_attention", "channels": 128, "reduction_ratio": 16},
        {"type": "conv", "in_channels": 128, "out_channels": 256, "kernel_size": 3},
        {"type": "pool", "output_size": 1, "pool_type": "avg"},
    ],
    "classifier": {"in_features": 256, "num_classes": 10}
}

model = ModelBuilder.from_config(config)
```

### Approach 3: Compose from Components

```python
from src.ml.supervised.models.pytorch.base_model import PyTorchBaseModel
from src.ml.supervised.models.pytorch.components import (
    ConvBlock, ResBlock, SelfAttention, AttentionPooling
)

class MyResearchModel(PyTorchBaseModel):
    def __init__(self, num_classes=10):
        super().__init__()

        # Mix and match components
        self.conv1 = ConvBlock(1, 64, activation='gelu')
        self.res1 = ResBlock(64)
        self.conv2 = ConvBlock(64, 128, activation='gelu')
        self.attention = SelfAttention(embed_dim=128, num_heads=4)
        self.pool = AttentionPooling(input_dim=128)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.conv2(x)

        # Reshape for attention: (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.attention(x)
        x = x.mean(dim=1)  # Average over sequence

        return self.fc(x)

    def get_config(self):
        return {"model_class": self.__class__.__name__}

# Register it!
ModelRegistry.register("my_research_model", MyResearchModel)
```

### Approach 4: Quick Config Generation

```python
from src.ml.supervised.models.pytorch.builder import create_simple_cnn_config

# Generate baseline config
config = create_simple_cnn_config(
    input_channels=1,
    num_classes=10,
    hidden_dims=[32, 64, 128],
)

model = ModelBuilder.from_config(config)
```

---

## 📦 Data Pipeline

### Quick Start: SimplePipeline

```python
from src.ml.supervised.data.pipeline import SimplePipeline
from src.ml.supervised.data.transforms import TransformPipeline
from src.ml.supervised.data.transforms.audio import AudioToMelSpectrogram
from src.ml.supervised.data.transforms.normalization import StandardNormalization

# Define transforms
transform = TransformPipeline([
    AudioToMelSpectrogram(n_mels=128),
    StandardNormalization(),
])

# Option 1: From folder structure (data/train/, data/val/, data/test/)
pipeline = SimplePipeline.from_folder(
    data_dir="data/audio",
    transform=transform,
    batch_size=32,
    num_workers=4
)

# Option 2: From CSV file
pipeline = SimplePipeline.from_csv(
    data_dir="data/audio",
    csv_file="data/labels.csv",
    transform=transform,
    batch_size=32
)

# Use loaders
train_loader = pipeline.train_loader
val_loader = pipeline.val_loader

for audio, labels in train_loader:
    # audio: (batch_size, channels, freq, time)
    # labels: (batch_size,)
    pass
```

### Advanced: DataPipeline with Custom Configuration

```python
from src.ml.supervised.data.pipeline import DataPipeline
from src.ml.supervised.data.loaders import PadCollate

# Create pipeline with custom settings
pipeline = DataPipeline(
    data_dir="data/audio",
    csv_file="data/train.csv",
    dataset_type="csv",
    transform=transform,
    sample_rate=22050,
    max_duration=5.0,  # Limit to 5 seconds
    batch_size=64,
    num_workers=8,
    collate_fn=PadCollate(max_length=110250)  # Handle variable lengths
)

# Get loaders
train_loader = pipeline.get_train_loader()
val_loader = pipeline.get_val_loader()
test_loader = pipeline.get_test_loader()

# View summary
print(pipeline.summary())
```

### Dataset Options

#### 1. CSV-Based Dataset

```python
from src.ml.supervised.data.datasets import AudioClassificationDataset

# CSV format:
#   file_path,label
#   audio/train/song1.wav,0
#   audio/train/song2.wav,1

dataset = AudioClassificationDataset(
    data_dir="data/audio",
    csv_file="data/train.csv",
    split="train",
    transform=transform,
    sample_rate=22050,
    max_duration=10.0,
    label_map={"rock": 0, "jazz": 1, "classical": 2}  # Optional string->int mapping
)
```

#### 2. Folder-Based Dataset

```python
from src.ml.supervised.data.datasets import FolderAudioDataset

# Folder structure:
#   data/train/rock/*.wav
#   data/train/jazz/*.wav
#   data/val/rock/*.wav

dataset = FolderAudioDataset(
    data_dir="data",
    split="train",
    transform=transform,
    sample_rate=22050,
    audio_extensions=(".wav", ".mp3", ".flac")
)

print(f"Classes: {dataset.class_names}")
print(f"Samples: {len(dataset)}")
```

#### 3. In-Memory Dataset (Fast Iteration)

```python
from src.ml.supervised.data.datasets import InMemoryAudioDataset

# Preload all audio for faster training
dataset = InMemoryAudioDataset(
    data_dir="data/audio",
    csv_file="data/train.csv",
    split="train",
    transform=transform,
    preload=True  # Load all audio at initialization
)
```

### Custom Collate Functions

```python
from src.ml.supervised.data.loaders import (
    PadCollate,
    FixedLengthCollate,
    SpectrogramCollate
)

# Pad variable-length audio
collate_fn = PadCollate(
    max_length=100000,
    pad_value=0.0,
    return_lengths=True
)

# Fixed-length cropping/padding
collate_fn = FixedLengthCollate(
    length=88200,  # 4 seconds at 22050 Hz
    crop_mode="random"  # 'start', 'center', or 'random'
)

# Spectrogram-specific padding
collate_fn = SpectrogramCollate(
    max_time_frames=500,
    return_lengths=False
)

# Use with DataLoader
from src.ml.supervised.data.loaders import DataLoaderFactory

loader = DataLoaderFactory.create_train_loader(
    dataset=dataset,
    batch_size=32,
    num_workers=4,
    collate_fn=collate_fn
)
```

### Complete Example: Custom Pipeline

```python
from src.ml.supervised.data.pipeline import DataPipeline
from src.ml.supervised.data.transforms import TransformPipeline
from src.ml.supervised.data.transforms.audio import AudioToMelSpectrogram
from src.ml.supervised.data.transforms.augmentation import SpecAugment
from src.ml.supervised.data.transforms.normalization import DBScaling
from src.ml.supervised.data.loaders import FixedLengthCollate

# Preprocessing pipeline
transform = TransformPipeline([
    AudioToMelSpectrogram(
        sample_rate=22050,
        n_mels=128,
        fmax=8000
    ),
    SpecAugment(
        time_mask_max=10,
        freq_mask_max=8,
        num_time_masks=2
    ),
    DBScaling(top_db=80),
])

# Data pipeline
pipeline = DataPipeline(
    data_dir="data/music",
    csv_file="data/metadata.csv",
    dataset_type="csv",
    transform=transform,
    sample_rate=22050,
    max_duration=30.0,
    batch_size=32,
    num_workers=8,
    collate_fn=FixedLengthCollate(length=660000, crop_mode="random")
)

# Get loaders
train_loader = pipeline.get_train_loader()
val_loader = pipeline.get_val_loader(batch_size=64)  # Override batch size for val

# Training loop
for epoch in range(num_epochs):
    for audio, labels in train_loader:
        # audio: (32, 1, 128, time_frames)
        # labels: (32,)

        outputs = model(audio)
        loss = criterion(outputs, labels)
        # ...
```

---

## 🔬 Research Workflow Examples

### Experiment 1: Baseline

```python
# Data preprocessing
pipeline = TransformPipeline([
    AudioToMelSpectrogram(n_mels=128),
    StandardNormalization(),
])

# Model
model = ModelRegistry.create("simple_cnn", num_classes=10)

# Train...
```

### Experiment 2: Test Augmentation Impact

```python
# Add augmentation
pipeline = TransformPipeline([
    AudioToMelSpectrogram(n_mels=128),
    SpecAugment(time_mask_max=10, freq_mask_max=8),  # NEW
    StandardNormalization(),
])

# Same model
model = ModelRegistry.create("simple_cnn", num_classes=10)
```

### Experiment 3: Try Attention

```python
# Same preprocessing
pipeline = TransformPipeline([
    AudioToMelSpectrogram(n_mels=128),
    SpecAugment(time_mask_max=10, freq_mask_max=8),
    StandardNormalization(),
])

# Attention model
model = ModelRegistry.create("attention_cnn", num_classes=10)  # NEW
```

### Experiment 4: Custom Architecture

```python
# Try MFCC instead
pipeline = TransformPipeline([
    AudioToMFCC(n_mfcc=13),
    RobustNormalization(),
])

# Custom architecture with more depth
config = {
    "blocks": [
        {"type": "conv", "in_channels": 1, "out_channels": 32},
        {"type": "resblock", "channels": 32},
        {"type": "conv", "in_channels": 32, "out_channels": 64},
        {"type": "resblock", "channels": 64},
        {"type": "conv", "in_channels": 64, "out_channels": 128},
        {"type": "channel_attention", "channels": 128},
        {"type": "pool", "output_size": 1},
    ],
    "classifier": {"in_features": 128, "num_classes": 10}
}

model = ModelBuilder.from_config(config)
```

---

## 📦 Available Model Components

### Convolutional Blocks
- `ConvBlock` - Conv + BN + Activation + Dropout
- `ResBlock` - Residual block with skip connection
- `DepthwiseSeparableConv` - Efficient MobileNet-style conv

### Attention Mechanisms
- `SelfAttention` - Multi-head self-attention
- `MultiHeadAttention` - PyTorch native implementation
- `ChannelAttention` - Squeeze-and-excitation style

### Pooling Strategies
- `AdaptivePooling` - Avg/Max/Both pooling
- `AttentionPooling` - Learnable weighted pooling
- `GlobalPooling` - Avg/Max/Std pooling

---

## 🎯 Tips for Research

1. **Start Simple**: Begin with `simple_cnn` and basic transforms
2. **Change One Thing**: Modify one component at a time to understand impact
3. **Log Everything**: Use wandb/mlflow to track all experiments
4. **Create Variants**: Easy to create model variants with registry
5. **Custom Transforms**: Subclass `BaseTransform` for your ideas

---

## 📝 Example: Complete Experiment Script

```python
# experiments/my_experiment.py
import torch
import torch.nn as nn
from src.ml.supervised.data.pipeline import SimplePipeline
from src.ml.supervised.data.transforms import TransformPipeline
from src.ml.supervised.data.transforms.audio import AudioToMelSpectrogram
from src.ml.supervised.data.transforms.augmentation import SpecAugment
from src.ml.supervised.data.transforms.normalization import DBScaling
from src.ml.supervised.models.pytorch.registry import ModelRegistry
from src.ml.shared.logging.factory import LoggerFactory

# 1. Setup preprocessing
transform = TransformPipeline([
    AudioToMelSpectrogram(sample_rate=22050, n_mels=128, fmax=8000),
    SpecAugment(time_mask_max=10, freq_mask_max=8, num_time_masks=2),
    DBScaling(top_db=80),
])

# 2. Create data pipeline
pipeline = SimplePipeline.from_folder(
    data_dir="data/audio",
    transform=transform,
    batch_size=32,
    num_workers=4
)

# 3. Create model
model = ModelRegistry.create("attention_cnn", num_classes=10, hidden_dims=[64, 128, 256])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 4. Setup training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Setup logging
logger = LoggerFactory.create_from_name("wandb", "music-research", "exp-attention-v1")
logger.log_params({
    "model": "attention_cnn",
    "transforms": str(transform),
    "optimizer": "adam",
    "learning_rate": 0.001,
    "batch_size": 32,
})

# 6. Train
num_epochs = 50
for epoch in range(num_epochs):
    # Training
    model.train()
    for audio, labels in pipeline.train_loader:
        audio, labels = audio.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(audio)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        logger.log_metrics({"train_loss": loss.item()})

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for audio, labels in pipeline.val_loader:
            audio, labels = audio.to(device), labels.to(device)
            outputs = model(audio)
            val_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = correct / len(pipeline.val_loader.dataset)
    logger.log_metrics({"val_loss": val_loss, "val_acc": val_acc})
    print(f"Epoch {epoch}: val_acc={val_acc:.3f}")

# 7. Done
logger.finish()
```

---

## 🚀 Next Steps

- Explore `components/` for building blocks
- Check `architectures/` for example models
- Read `builder.py` to understand config format
- Experiment with different transform combinations
