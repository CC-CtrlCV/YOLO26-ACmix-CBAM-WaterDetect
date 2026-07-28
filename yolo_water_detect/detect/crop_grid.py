from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Tile:
    image: np.ndarray
    x1: int
    y1: int
    x2: int
    y2: int


def split_grid(image: np.ndarray, rows: int = 4, cols: int = 6, overlap: float = 0.25) -> list[Tile]:
    """Split an image into overlapped tiles for small-object inference."""
    if image is None or image.size == 0:
        return []
    h, w = image.shape[:2]
    tile_w = max(1, int(np.ceil(w / cols)))
    tile_h = max(1, int(np.ceil(h / rows)))
    step_x = max(1, int(tile_w * (1.0 - overlap)))
    step_y = max(1, int(tile_h * (1.0 - overlap)))
    tiles: list[Tile] = []
    y = 0
    while y < h:
        x = 0
        y2 = min(h, y + tile_h)
        y1 = max(0, y2 - tile_h)
        while x < w:
            x2 = min(w, x + tile_w)
            x1 = max(0, x2 - tile_w)
            tiles.append(Tile(image=image[y1:y2, x1:x2].copy(), x1=x1, y1=y1, x2=x2, y2=y2))
            if x2 >= w:
                break
            x += step_x
        if y2 >= h:
            break
        y += step_y
    return tiles


def draw_grid(
    image: np.ndarray, rows: int = 4, cols: int = 6, color: tuple[int, int, int] = (255, 180, 0)
) -> np.ndarray:
    """Draw a simple rows x cols grid on a BGR image."""
    import cv2

    out = image.copy()
    h, w = out.shape[:2]
    for c in range(1, cols):
        x = int(w * c / cols)
        cv2.line(out, (x, 0), (x, h), color, 1, cv2.LINE_AA)
    for r in range(1, rows):
        y = int(h * r / rows)
        cv2.line(out, (0, y), (w, y), color, 1, cv2.LINE_AA)
    return out
