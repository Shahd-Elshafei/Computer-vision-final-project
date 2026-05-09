import cv2
import numpy as np
import sys
import os

from filters  import apply_blur, apply_sharpen
from blending import alpha_blend, save_result



def _get_color_mask(image_bgr, lower_hsv, upper_hsv,
                    blur_kernel=(5, 5), morph_kernel_size=5, feather_sigma=3):
    """
    Creates a refined, feathered binary mask for a given HSV color range.

    Args:
        image_bgr         : Input image in BGR format.
        lower_hsv         : Lower bound of target color (numpy array).
        upper_hsv         : Upper bound of target color (numpy array).
        blur_kernel       : Pre-blur kernel to reduce noise before masking.
        morph_kernel_size : Size of kernel for morphological cleanup.
        feather_sigma     : Sigma for Gaussian feathering on mask edges.

    Returns:
        mask_float : Float32 mask [0.0, 1.0] shaped (H, W, 1).
    """
    # Pre-blur to reduce color noise before thresholding
    blurred = cv2.GaussianBlur(image_bgr, blur_kernel, 0)

    # Convert to HSV — lighting-invariant color space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Threshold to get raw binary mask
    raw_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Morphological cleanup : remove speckles and fill holes
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    # Feather edges for a natural, smooth transition
    feathered = cv2.GaussianBlur(clean_mask, (0, 0), feather_sigma)

    # Normalize to float and add channel dimension for blending
    mask_float = (feathered / 255.0).astype(np.float32)
    return mask_float[:, :, np.newaxis]


def _get_color_mask_multi_range(image_bgr, hsv_ranges, **kwargs):
    """
    Handles colors that wrap around the HSV hue wheel (like red spans
    0-15 AND 165-180 degrees) + Merges multiple range masks with union.

    Args:
        image_bgr  : Input image in BGR format.
        hsv_ranges : List of (lower_hsv, upper_hsv) tuples.
        **kwargs   : Passed through to _get_color_mask.

    Returns:
        Combined float mask (H, W, 1).
    """
    h, w = image_bgr.shape[:2]
    combined = np.zeros((h, w, 1), dtype=np.float32)
    for lower, upper in hsv_ranges:
        mask = _get_color_mask(image_bgr, lower, upper, **kwargs)
        combined = np.maximum(combined, mask)
    return combined



def selective_color_enhancement(image_path, output_path,
                                 target_color="green",
                                 sharpen_strength=1.5,
                                 blur_radius=25,
                                 mask_output_path=None):
    """
    Enhances a target-colored object (sharpen) while blurring
    the rest of the image 

    Args:
        image_path       : Path to the input image file.
        output_path      : Path to save the output image.
        target_color     : Color region to sharpen: "green", "red", "yellow".
        sharpen_strength : Intensity of sharpening on the target region.
        blur_radius      : Strength of blur applied to non-target region.
        mask_output_path : path to save the generated mask.

    """

    # Step 1: Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from: {image_path}")
    print(f"Image loaded: {image.shape[1]}x{image.shape[0]} px")

    # Step 2: Define HSV color ranges per target color
    color_ranges = {
        # Green: hue 35-85 degrees (I have defined multiple ranges for the green to handle green color in nature better)
       "green": [
          (np.array([30, 30, 30]),   np.array([50, 255, 255])),  # Yellow-greens
           (np.array([50, 30, 30]),   np.array([85, 255, 255])),  # True greens
           (np.array([85, 30, 30]),   np.array([95, 255, 255])),  # Blue-greens
       ],

        # Yellow: hue 20-35 degrees
        "yellow": [
            (np.array([20, 80, 80]),   np.array([35, 255, 255]))
        ],
        # Red (needs two ranges)
        "red": [
            (np.array([0,  80, 80]),   np.array([10, 255, 255])),
            (np.array([165, 80, 80]),  np.array([180, 255, 255]))
        ]
    }

    if target_color not in color_ranges:
        raise ValueError(
            f"Unknown color '{target_color}'. "
            f"Choose from: {list(color_ranges.keys())}"
        )

    # Step 3: Build the color selection mask
    mask = _get_color_mask_multi_range(
        image,
        hsv_ranges=color_ranges[target_color],
        feather_sigma=5
    )

    # Save mask
    if mask_output_path:
        mask_display = (mask * 255).astype(np.uint8)
        cv2.imwrite(mask_output_path, mask_display)
        print(f"Saved mask: {mask_output_path}")

    # Step 4: Apply effects to each region separately using already built functions
    sharpened_layer = apply_sharpen(image, strength=sharpen_strength)
    blurred_layer   = apply_blur(image, blur_radius=blur_radius)

    # Step 5: Blend: sharp where mask=1, blurred where mask=0
    result = alpha_blend(
        foreground=sharpened_layer,
        background=blurred_layer,
        mask_float=mask
    )

    # Step 6: Save and return
    save_result(result, output_path)
    return result
