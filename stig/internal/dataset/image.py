"""
Package image provides image processing utilities.
"""

from typing import Tuple

import cv2, numpy as np, torch


def process_from_bytes(
    img_bytes: bytes,
    frame_size: Tuple[int, int],
    device: str = "cpu",
    grayscale: bool = False,
) -> np.ndarray:
    """
    Process an image for model inference.
    """
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), flag)
    if img is None:
        raise ValueError("Failed to decode image")

    if not grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = resize_image(
        img,
        height=frame_size[0],
        width=frame_size[1],
    )

    return img


def process_from_path(
    image_path: str,
    frame_size: Tuple[int, int],
    grayscale: bool = False,
) -> np.ndarray:
    """
    Process an image from file path for dataset building.
    """
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(image_path), flag)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")

    if not grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = resize_image(
        img,
        height=frame_size[0],
        width=frame_size[1],
    )

    return img


def resize_image(
    img: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Resize an image to the specified size.
    """
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def to_tensor(
    img: np.ndarray,
    device: str,
) -> torch.Tensor:
    """
    Convert a numpy image array to a PyTorch tensor:
    - grayscale (H, W) -> (1, 1, H, W) uint8
    - RGB (H, W, 3)    -> (1, 3, H, W) uint8
    """
    if img.ndim == 2:
        return torch.from_numpy(img)[None, None].to(device)
    return torch.from_numpy(img).permute(2, 0, 1)[None].contiguous().to(device)
