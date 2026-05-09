import cv2
import numpy as np
import matplotlib.pyplot as plt

def stitch_images(left_path, right_path):

    # 1. Load images
    img1 = cv2.imread(left_path)
    img2 = cv2.imread(right_path)

    # Show BEFORE images-
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title("Left Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title("Right Image")
    plt.axis("off")

    plt.show()

    # 2. Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 3. ORB feature detection
    orb = cv2.ORB_create(3000)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # 4. Feature matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)


    # 5. Extract matched points
    pts1 = np.float32(
        [kp1[m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    pts2 = np.float32(
        [kp2[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    # 6. Compute homography
    H, _ = cv2.findHomography(
        pts2,
        pts1,
        cv2.RANSAC,
        5.0
    )

    # 7. Stitch images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Corners of img2
    corners_img2 = np.float32([
        [0,0],
        [0,h2],
        [w2,h2],
        [w2,0]
    ]).reshape(-1,1,2)

    # Transform corners
    transformed_corners = cv2.perspectiveTransform(corners_img2, H)

    # Corners of img1
    corners_img1 = np.float32([
        [0,0],
        [0,h1],
        [w1,h1],
        [w1,0]
    ]).reshape(-1,1,2)

    # Combine all corners
    all_corners = np.concatenate(
        (corners_img1, transformed_corners),
        axis=0
    )

    [x_min, y_min] = np.int32(
        all_corners.min(axis=0).ravel()
    )

    [x_max, y_max] = np.int32(
        all_corners.max(axis=0).ravel()
    )

    # Translation
    translation = [-x_min, -y_min]

    H_translation = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])

    # Warp image
    result = cv2.warpPerspective(
        img2,
        H_translation @ H,
        (x_max - x_min, y_max - y_min)
    )

    # Place first image
    result[
        translation[1]:h1 + translation[1],
        translation[0]:w1 + translation[0]
    ] = img1


    # Show AFTER result
    plt.figure(figsize=(15,8))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Final Panorama")
    plt.show()

    return result
