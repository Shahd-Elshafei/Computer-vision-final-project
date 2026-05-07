import cv2
import numpy as np


def apply_blur(image_bgr, blur_radius=21):
    """
    Applies Gaussian blur to an image.
    Args:
        image_bgr   : Input image in BGR format.
        blur_radius : Kernel size — must be odd. Higher = more blurred.

    Returns:
        Blurred image as uint8 BGR.
    """
    if blur_radius % 2 == 0:
        blur_radius += 1

    # 0: Tells OpenCV to automatically calculate sigma (standard deviation) from kernel size
    return cv2.GaussianBlur(image_bgr, (blur_radius, blur_radius), 0)


def apply_sharpen(image_bgr, strength=1.5):
    """
    Sharpens an image using the Unsharp Mask technique (/ high boost).

    Args:
        image_bgr : Input image in BGR format.
        strength  : Sharpening intensity. 1.0 = subtle, 2.0+ = aggressive.

    Returns:
        Sharpened image as uint8 BGR.
    """
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=3)
    # (0, 0) means: "OpenCV, please figure out the best kernel size automatically"
    # The kernel size is calculated as: size = 2 × ceil(2 × sigma) + 1
    # sigmaX=3 controls what details get extracted

    sharpened = cv2.addWeighted(image_bgr, 1 + strength, blurred, -strength, 0)
    # addWeighted(source1, alpha, source2, beta, gamma)
    # Result = source1 × alpha + source2 × beta + gamma
    # sharpened = original × (1 + strength) + blurred × (-strength) + 0




    return sharpened
