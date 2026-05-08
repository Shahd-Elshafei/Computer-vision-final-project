import cv2
import numpy as np
import sys
import os
from blending import save_result


def transformed_object_recognition(object_path, scene_path, output_path,
                                    min_match_count=10,
                                    draw_lines=True):  
    """
    Detects an object in a scene image even if rotated, scaled, or translated.
    Uses SIFT keypoints + FLANN matching + RANSAC homography.

    Args:
        object_path     : Path to the clean object image .
        scene_path      : Path to the scene image after adjustements.
        output_path     : Path to save the result visualization.
        min_match_count : Minimum good matches needed to confirm detection.
        draw_lines      : If True, shows match lines between images.
                          If False, shows only bounding box on scene (clean output).
    """

    # Step 1: Load both images
    img_object = cv2.imread(object_path, cv2.IMREAD_GRAYSCALE)
    img_scene  = cv2.imread(scene_path,  cv2.IMREAD_GRAYSCALE)

    if img_object is None:
        raise FileNotFoundError(f"Could not load object image: {object_path}")
    if img_scene is None:
        raise FileNotFoundError(f"Could not load scene image: {scene_path}")

    print(f"Object image : {img_object.shape[1]}x{img_object.shape[0]} px")
    print(f"Scene image  : {img_scene.shape[1]}x{img_scene.shape[0]} px")

    # Step 2: Detect SIFT keypoints and compute descriptors
    sift = cv2.SIFT_create()

    keypoints_obj,   descriptors_obj   = sift.detectAndCompute(img_object, None)
    keypoints_scene, descriptors_scene = sift.detectAndCompute(img_scene,  None)

    print(f"Keypoints found — Object: {len(keypoints_obj)}, Scene: {len(keypoints_scene)}")

    # Step 3: Match descriptors using FLANN (Fast Library for Approximate Nearest Neighbors)
    FLANN_INDEX_KDTREE = 1
    index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann   = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_obj, descriptors_scene, k=2) # For each object keypoint it finds the 2 best matching scene keypoints

    # Step 4: Lowe's ratio test — filter out ambiguous matches 
    # A match is good only if it's clearly better than the second-best match
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"Good matches after ratio test: {len(good_matches)}")

    # Step 5: Find homography if enough good matches exist 
    if len(good_matches) >= min_match_count:

        # Extract the matched keypoint coordinates from both images
        pts_obj   = np.float32(
            [keypoints_obj[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        pts_scene = np.float32(
            [keypoints_scene[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        # Compute homography where RANSAC((Random Sample Consensus)) handles outlier matches automatically
        H, mask = cv2.findHomography(pts_obj, pts_scene, cv2.RANSAC, 5.0)

        inliers = int(mask.sum())
        print(f"Homography found — Inlier matches: {inliers}")

        # Step 6: Draw bounding box around detected object in scene
        h, w = img_object.shape

        # The 4 corners of the object image
        corners_obj = np.float32([
            [0, 0], [w, 0], [w, h], [0, h]
        ]).reshape(-1, 1, 2)

        # Transform corners into scene coordinate space using homography
        corners_scene = cv2.perspectiveTransform(corners_obj, H)

        # Draw the bounding box on a color copy of the scene
        img_scene_color = cv2.cvtColor(img_scene, cv2.COLOR_GRAY2BGR)
        cv2.polylines(
            img_scene_color,
            [np.int32(corners_scene)],
            isClosed=True,
            color=(0, 255, 0),   # green box
            thickness=3
        )

        # Step 7: Draw match lines between object and scene (if draw_lines=True)
        if draw_lines:

            img_object_color = cv2.cvtColor(img_object, cv2.COLOR_GRAY2BGR)

            # Only draw inlier matches (mask=1) for a clean visualization
            draw_params = dict(
                matchColor=(0, 255, 0),       # green lines for good matches
                singlePointColor=(255, 0, 0), # blue for unmatched keypoints
                matchesMask=mask.ravel().tolist(),
                flags=cv2.DrawMatchesFlags_DEFAULT
            )

            result = cv2.drawMatches(
                img_object_color, keypoints_obj,
                img_scene_color,  keypoints_scene,
                good_matches, None, **draw_params
            )
        else:
            # Clean mode: just the scene with bounding box (no lines, no object image)
            result = img_scene_color
            print("Clean mode: Showing only bounding box")

        print("Object successfully detected in scene!")

    else:
        # Not enough matches case
        print(f"Detection failed : only {len(good_matches)} good matches "
              f"(need at least {min_match_count})")

        img_object_color = cv2.cvtColor(img_object, cv2.COLOR_GRAY2BGR)
        img_scene_color  = cv2.cvtColor(img_scene,  cv2.COLOR_GRAY2BGR)

        if draw_lines:
            result = cv2.drawMatches(
                img_object_color, keypoints_obj,
                img_scene_color,  keypoints_scene,
                good_matches, None,
                flags=cv2.DrawMatchesFlags_DEFAULT
            )
        else:
            # In clean mode, just show the scene even if detection failed
            result = img_scene_color
            print("Clean mode: Detection failed, showing scene without box")

    # Step 8: Save and return
    save_result(result, output_path)
    return result
