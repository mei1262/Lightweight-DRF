"""
Robustness disruption utilities for MVSA image-text sentiment analysis.

Reference: DRF paper (ACM MM 2024)
- Low-quality image: randomly mask 40-80% pixels
- Low-quality text: replace 40-80% words with [MASK]
- Missing modality: discard image or text (replaced with blank / [MASK] padding)
"""

import random
from enum import Enum
from typing import Optional, Tuple

import torch


class DisruptionType(str, Enum):
    """C=corruption only, D=discard/missing only, C_D=half corruption + half discard."""

    C = "C"
    D = "D"
    C_D = "C+D"


class DisruptionStrategy(str, Enum):
    FIXED = "fixed"
    RANDOM = "random"


def sample_corruption_ratio(
    ratio_range: Tuple[float, float] = (0.4, 0.8),
    rng: Optional[random.Random] = None,
) -> float:
    rng = rng or random
    low, high = ratio_range
    return rng.uniform(low, high)


def corrupt_image(
    image: torch.Tensor,
    mask_ratio: float,
    fill_value: float = 0.0,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    """
    Randomly mask ``mask_ratio`` of spatial pixels (all channels).

    Args:
        image: [C, H, W] tensor after normalization.
        mask_ratio: fraction of pixels to corrupt, in [0, 1].
    """
    if mask_ratio <= 0:
        return image

    rng = rng or random
    corrupted = image.clone()
    _, height, width = corrupted.shape
    num_pixels = height * width
    num_mask = int(num_pixels * mask_ratio)

    if num_mask <= 0:
        return corrupted

    flat_indices = rng.sample(range(num_pixels), num_mask)
    for idx in flat_indices:
        row, col = divmod(idx, width)
        corrupted[:, row, col] = fill_value

    return corrupted


def corrupt_text(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_token_id: int,
    mask_ratio: float,
    pad_token_id: int,
    rng: Optional[random.Random] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replace ``mask_ratio`` of valid (non-pad) tokens with [MASK].
    """
    if mask_ratio <= 0:
        return input_ids.clone(), attention_mask.clone()

    rng = rng or random
    corrupted_ids = input_ids.clone()
    corrupted_mask = attention_mask.clone()

    valid_indices = [
        idx for idx, mask_val in enumerate(attention_mask.tolist()) if mask_val == 1
    ]
    if not valid_indices:
        return corrupted_ids, corrupted_mask

    num_mask = max(1, int(len(valid_indices) * mask_ratio))
    num_mask = min(num_mask, len(valid_indices))
    mask_indices = rng.sample(valid_indices, num_mask)

    for idx in mask_indices:
        corrupted_ids[idx] = mask_token_id

    return corrupted_ids, corrupted_mask


def make_missing_image(image: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    """Blank image used when the visual modality is discarded."""
    return torch.full_like(image, fill_value)


def make_missing_text(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """All-[MASK] text used when the textual modality is discarded."""
    missing_ids = torch.full_like(input_ids, mask_token_id)
    missing_mask = attention_mask.clone()
    return missing_ids, missing_mask


def decide_disruption(
    sample_index: int,
    strategy: DisruptionStrategy,
    disruption_type: DisruptionType,
    disruption_ratio: float,
    disrupted_modality: str,
    rng: random.Random,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Decide whether/how to disrupt one sample.

    Returns:
        (is_disrupted, target_modality, operation)
        operation in {"corrupt", "discard", None}
    """
    if strategy == DisruptionStrategy.FIXED:
        target = disrupted_modality
        if disruption_type == DisruptionType.C:
            return True, target, "corrupt"
        if disruption_type == DisruptionType.D:
            return True, target, "discard"
        # C+D: half corruption, half discard on the fixed modality
        operation = "corrupt" if sample_index % 2 == 0 else "discard"
        return True, target, operation

    # modality-random: only ``disruption_ratio`` samples are disrupted
    if rng.random() >= disruption_ratio:
        return False, None, None

    target = rng.choice(["image", "text"])
    if disruption_type == DisruptionType.C:
        return True, target, "corrupt"
    if disruption_type == DisruptionType.D:
        return True, target, "discard"

    operation = "corrupt" if rng.random() < 0.5 else "discard"
    return True, target, operation
