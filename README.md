# Computer-vision-final-project
# Object Recognition & Color Enhancement Module

> **Note**: This document covers only **Task 1** and **Task 6** of an 8-task computer vision project. The full project includes additional work by other team members.

---

## Task 1: Selective Color Enhancement

### Purpose

Enhances a target-colored object (sharpen it) while blurring the rest of the image. Useful for highlighting specific colored objects in a scene.

### How It Works

1. Loads an input image
2. Defines HSV color ranges for target colors (green, red, yellow)
3. Creates a binary mask isolating the target color
4. Applies sharpening to the masked region
5. Applies blurring to everything outside the mask
6. Blends both layers into a single output image

### Key Functions

| Function | Purpose |
|----------|---------|
| `_get_color_mask()` | Creates feathered HSV mask |
| `_get_color_mask_multi_range()` | Handles colors that wrap around HSV (e.g., red) |
| `selective_color_enhancement()` | Main function |

### Supported Colors

- **Green** – Multiple ranges (yellow-greens, true greens, blue-greens)
- **Red** – Two ranges (0-10° and 165-180°)
- **Yellow** – Single range (20-35°)

### Output

- Enhanced image with sharp target object + blurred background
- Optional mask export

---

## Task 6: Transformed Object Recognition

### Purpose

Detects a specific object in a scene even when the object is **rotated, scaled, or translated**. Draws a bounding box around the detected object.

### How It Works

1. Loads object image (clean reference) and scene image
2. Detects SIFT keypoints and computes descriptors for both images
3. Matches descriptors using FLANN (fast approximate nearest neighbors)
4. Applies Lowe's ratio test to filter ambiguous matches
5. Computes homography using RANSAC to handle outliers
6. Transforms object corners into scene coordinates
7. Draws bounding box (and optional match lines) on the scene

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_match_count` | 10 | Minimum good matches to confirm detection |
| `draw_lines` | True | Shows match lines (False = clean output) |


### Output

- Scene image with green bounding box around detected object
- Optional visualization with match lines between object and scene

---

## Dependencies

```bash
pip install opencv-python numpy
