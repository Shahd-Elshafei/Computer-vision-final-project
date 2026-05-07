import cv2
import numpy as np


def alpha_blend(foreground, background, mask_float):
    """
    Blends two images using a soft float mask via alpha compositing.
    Formula: result = mask * foreground + (1 - mask) * background
    Shared across: Task 1 (selective blend), Task 3 (enhancement blend),
                   Task 5 (panorama seam blend), Task 8 (HDR exposure merge)

    Args:
        foreground  : Image shown where mask = 1.0 (the enhanced region).
        background  : Image shown where mask = 0.0 (the rest).
        mask_float  : Float32 mask shaped (H, W, 1), values in [0.0, 1.0].

    Returns:
        Blended uint8 BGR image.
    """
    fg = foreground.astype(np.float32)
    bg = background.astype(np.float32)
    blended = mask_float * fg + (1.0 - mask_float) * bg
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_result(image_bgr, output_path):
    """
    Saves the final processed image to disk.

    Args:
        image_bgr   : Final image in BGR format.
        output_path : Full path including filename and extension.
    """
    success = cv2.imwrite(output_path, image_bgr)
    if not success:
        raise IOError(f"Failed to write image to: {output_path}")
    print(f"Saved: {output_path}")
