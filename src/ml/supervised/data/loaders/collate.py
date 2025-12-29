"""Custom collate functions for batching audio data.

Handles variable-length audio sequences and different padding strategies.
"""

from typing import List, Tuple, Any
import torch
import numpy as np


class PadCollate:
    """Collate function that pads sequences to max length in batch.

    Research use: Handle variable-length audio clips in batches.

    Args:
        max_length: Maximum sequence length (clips longer sequences)
        pad_value: Value to use for padding
        return_lengths: Whether to return sequence lengths

    Example:
        >>> from src.ml.supervised.data.loaders import PadCollate
        >>>
        >>> collate_fn = PadCollate(max_length=100000)
        >>> loader = DataLoader(dataset, collate_fn=collate_fn)
    """

    def __init__(
        self,
        max_length: int | None = None,
        pad_value: float = 0.0,
        return_lengths: bool = False,
    ):
        self.max_length = max_length
        self.pad_value = pad_value
        self.return_lengths = return_lengths

    def __call__(
        self, batch: List[Tuple[np.ndarray | torch.Tensor, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Collate batch with padding.

        Args:
            batch: List of (audio, label) tuples

        Returns:
            Padded audio batch, labels, and optionally lengths
        """
        audios, labels = zip(*batch)

        # Convert to tensors
        audios = [
            torch.from_numpy(a) if isinstance(a, np.ndarray) else a for a in audios
        ]
        labels = torch.tensor(labels)

        # Get lengths
        lengths = torch.tensor([len(a) for a in audios])

        # Determine max length
        if self.max_length is not None:
            max_len = min(self.max_length, max(lengths).item())
        else:
            max_len = max(lengths).item()

        # Pad sequences
        padded_audios = []
        for audio in audios:
            if len(audio) > max_len:
                # Clip if too long
                padded = audio[:max_len]
            elif len(audio) < max_len:
                # Pad if too short
                pad_size = max_len - len(audio)
                if audio.dim() == 1:
                    padded = torch.nn.functional.pad(
                        audio, (0, pad_size), value=self.pad_value
                    )
                else:
                    # Multi-dimensional (e.g., spectrograms)
                    padded = torch.nn.functional.pad(
                        audio, (0, pad_size), value=self.pad_value
                    )
            else:
                padded = audio

            padded_audios.append(padded)

        # Stack into batch
        padded_batch = torch.stack(padded_audios)

        if self.return_lengths:
            return padded_batch, labels, lengths

        return padded_batch, labels


class FixedLengthCollate:
    """Collate function that crops/pads to fixed length.

    Research use: Fixed-size inputs for models that don't handle variable lengths.

    Args:
        length: Fixed sequence length
        pad_value: Value to use for padding
        crop_mode: How to crop ('start', 'center', 'random')

    Example:
        >>> collate_fn = FixedLengthCollate(length=88200, crop_mode='random')
    """

    def __init__(
        self,
        length: int,
        pad_value: float = 0.0,
        crop_mode: str = "center",
    ):
        self.length = length
        self.pad_value = pad_value
        self.crop_mode = crop_mode

        if crop_mode not in ["start", "center", "random"]:
            raise ValueError(
                f"crop_mode must be 'start', 'center', or 'random', got {crop_mode}"
            )

    def __call__(
        self, batch: List[Tuple[np.ndarray | torch.Tensor, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate batch with fixed length.

        Args:
            batch: List of (audio, label) tuples

        Returns:
            Fixed-length audio batch and labels
        """
        audios, labels = zip(*batch)

        # Convert to tensors
        audios = [
            torch.from_numpy(a) if isinstance(a, np.ndarray) else a for a in audios
        ]
        labels = torch.tensor(labels)

        # Process each audio
        processed = []
        for audio in audios:
            if len(audio) > self.length:
                # Crop
                if self.crop_mode == "start":
                    cropped = audio[: self.length]
                elif self.crop_mode == "center":
                    start = (len(audio) - self.length) // 2
                    cropped = audio[start : start + self.length]
                else:  # random
                    start = torch.randint(0, len(audio) - self.length + 1, (1,)).item()
                    cropped = audio[start : start + self.length]
                processed.append(cropped)
            elif len(audio) < self.length:
                # Pad
                pad_size = self.length - len(audio)
                if audio.dim() == 1:
                    padded = torch.nn.functional.pad(
                        audio, (0, pad_size), value=self.pad_value
                    )
                else:
                    padded = torch.nn.functional.pad(
                        audio, (0, pad_size), value=self.pad_value
                    )
                processed.append(padded)
            else:
                processed.append(audio)

        # Stack into batch
        batch_tensor = torch.stack(processed)

        return batch_tensor, labels


class SpectrogramCollate:
    """Collate function for spectrograms with time-axis padding.

    Research use: Handle variable-length spectrograms.

    Args:
        max_time_frames: Maximum time frames (None = use batch max)
        pad_value: Value to use for padding
        return_lengths: Whether to return time lengths

    Example:
        >>> collate_fn = SpectrogramCollate(max_time_frames=500)
    """

    def __init__(
        self,
        max_time_frames: int | None = None,
        pad_value: float = 0.0,
        return_lengths: bool = False,
    ):
        self.max_time_frames = max_time_frames
        self.pad_value = pad_value
        self.return_lengths = return_lengths

    def __call__(
        self, batch: List[Tuple[np.ndarray | torch.Tensor, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Collate spectrograms with time-axis padding.

        Args:
            batch: List of (spectrogram, label) tuples
                   Spectrograms shape: (freq_bins, time_frames) or (channels, freq_bins, time_frames)

        Returns:
            Padded spectrogram batch, labels, and optionally lengths
        """
        spectrograms, labels = zip(*batch)

        # Convert to tensors
        spectrograms = [
            torch.from_numpy(s) if isinstance(s, np.ndarray) else s
            for s in spectrograms
        ]
        labels = torch.tensor(labels)

        # Get time lengths (last dimension)
        lengths = torch.tensor([s.shape[-1] for s in spectrograms])

        # Determine max time frames
        if self.max_time_frames is not None:
            max_frames = min(self.max_time_frames, max(lengths).item())
        else:
            max_frames = max(lengths).item()

        # Pad spectrograms
        padded_specs = []
        for spec in spectrograms:
            time_frames = spec.shape[-1]

            if time_frames > max_frames:
                # Crop
                padded = spec[..., :max_frames]
            elif time_frames < max_frames:
                # Pad on time axis
                pad_size = max_frames - time_frames
                if spec.dim() == 2:
                    # (freq, time)
                    padded = torch.nn.functional.pad(
                        spec, (0, pad_size), value=self.pad_value
                    )
                else:
                    # (channels, freq, time)
                    padded = torch.nn.functional.pad(
                        spec, (0, pad_size), value=self.pad_value
                    )
            else:
                padded = spec

            padded_specs.append(padded)

        # Stack into batch
        padded_batch = torch.stack(padded_specs)

        if self.return_lengths:
            return padded_batch, labels, lengths

        return padded_batch, labels


def default_collate(
    batch: List[Tuple[np.ndarray | torch.Tensor, Any]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Default collate function for fixed-size data.

    Research use: When all samples are already the same size.

    Args:
        batch: List of (data, label) tuples

    Returns:
        Stacked data batch and labels
    """
    data, labels = zip(*batch)

    # Convert to tensors
    data = [torch.from_numpy(d) if isinstance(d, np.ndarray) else d for d in data]
    labels = torch.tensor(labels)

    # Stack
    data_batch = torch.stack(data)

    return data_batch, labels
