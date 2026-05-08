import cv2
import numpy as np
import matplotlib.pyplot as plt

def xray_enhancement_pipeline(
    image_path,
    clahe_clip=2.0,
    clahe_grid=(8, 8),
    gamma=1.1,
    kernel_size=3,
    show=True
):
    # 1. Grayscale
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise ValueError("Invalid image path")

    img = original.astype(np.float32)

    # 2. Mean filtering
    img = cv2.blur(img, (kernel_size, kernel_size))

    # 3. CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    img = clahe.apply(img.astype(np.uint8)).astype(np.float32)

    # 4. Gamma correction
    img = (img / 255.0) ** gamma * 255.0

    # 5. sharpening
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    img = img + 0.4 * (img - blur)

    # 6. Normalization
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype(np.uint8)

    # Display
    if show:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("Original X-ray")
        plt.imshow(original, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Enhanced X-ray")
        plt.imshow(img, cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return original, img
  
def xray_frequency_pipeline(
    image_path,
    d0=40,
    alpha=0.6,
    amplification=2,
    show=True
):

    # 1. Grayscale
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if original is None:
        raise ValueError("Invalid image path")

    img = original.astype(np.float32)

    # 2. FFT
    f = np.fft.fft2(img)

    # 3. Shift spectrum to center
    fshift = np.fft.fftshift(f)

    # 4. Butterworth High-Pass Filter
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    H = np.zeros((rows, cols), dtype=np.float32)

    for u in range(rows):
        for v in range(cols):

            D = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)

            H[u, v] = 1 / (1 + (d0 / (D + 1e-5)) ** 4)

    # Balanced frequency enhancement
    fshift_filtered = (alpha * fshift) + ((1 - alpha) * fshift * H)

    # Frequency amplification
    fshift_filtered *= amplification

    # 5. Inverse FFT-
    f_ishift = np.fft.ifftshift(fshift_filtered)

    img = np.fft.ifft2(f_ishift)

    img = np.abs(img)

    # 6. Normalize final output
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    img = img.astype(np.uint8)

    # Display
    if show:

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("Original X-ray")
        plt.imshow(original, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Frequency Enhanced X-ray")
        plt.imshow(img, cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return original, img

def xray_hybrid_pipeline(
    image_path,
    d0=30,
    alpha=0.5,
    amp=1.5,
    clahe_clip=2.0,
    gamma=0.9,
    w_freq=0.5,
    show=True
):

    # 1. Grayscale
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if original is None:
        raise ValueError("Invalid image path")

    img = original.astype(np.float32)


    # SPATIAL BRANCH
    spatial = cv2.blur(img, (3, 3))

    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip,
        tileGridSize=(8, 8)
    )
    spatial = clahe.apply(spatial.astype(np.uint8)).astype(np.float32)

    spatial = (spatial / 255.0) ** gamma * 255.0

    blur = cv2.GaussianBlur(spatial, (3, 3), 0)
    spatial = spatial + 0.4 * (spatial - blur)

    spatial = cv2.normalize(spatial, None, 0, 255, cv2.NORM_MINMAX)


    #  FREQUENCY BRANCH
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    H = np.zeros((rows, cols), dtype=np.float32)

    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - crow)**2 + (v - ccol)**2)
            H[u, v] = 1 / (1 + (d0 / (D + 1e-5))**4)

    fshift_filtered = (alpha * fshift) + ((1 - alpha) * fshift * H)
    fshift_filtered *= amp

    img_freq = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    img_freq = np.abs(img_freq)

    img_freq = cv2.normalize(img_freq, None, 0, 255, cv2.NORM_MINMAX)

    #  HYBRID FUSION
    hybrid = (
        w_freq * img_freq +
        (1 - w_freq) * spatial
    )

    hybrid = np.clip(hybrid, 0, 255).astype(np.uint8)

    # DISPLAY
    if show:

        plt.figure(figsize=(12, 8))

        # 1. Original
        plt.subplot(2, 2, 1)
        plt.title("Original X-ray")
        plt.imshow(original, cmap="gray")
        plt.axis("off")

        # 2. Spatial enhancement
        plt.subplot(2, 2, 2)
        plt.title("Spatial Enhancement")
        plt.imshow(spatial.astype(np.uint8), cmap="gray")
        plt.axis("off")

        # 3. Frequency enhancement
        plt.subplot(2, 2, 3)
        plt.title("Frequency Enhancement")
        plt.imshow(img_freq.astype(np.uint8), cmap="gray")
        plt.axis("off")

        # 4. Hybrid result
        plt.subplot(2, 2, 4)
        plt.title("Hybrid Result")
        plt.imshow(hybrid, cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return original, spatial.astype(np.uint8), img_freq.astype(np.uint8), hybrid
