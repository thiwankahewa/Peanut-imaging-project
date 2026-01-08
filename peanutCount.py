#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np

def count_peanuts(img_gray, debug=False):
    """
    Input:
        img_gray : grayscale image (numpy array, 0–255)
    Returns:
        count           : estimated peanut count
        peanut_labels   : list of connected-component labels that are peanuts
        labels          : full label image (same size as input)
        stats           : connected component stats from OpenCV
    """

    # 1. Threshold – peanuts are bright, background dark
    #    Otsu automatically finds a good threshold value.
    _, thresh = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 2. Morphological opening to remove tiny noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 3. Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        clean, connectivity=8
    )

    # stats[0] is background; rows 1..num_labels-1 are objects
    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background

    # Remove very tiny blobs (dust, noise) before computing "typical" area
    # You can adjust 50 depending on your resolution.
    big_areas = areas[areas > 50]
    if len(big_areas) == 0:
        return 0, [], labels, stats

    # Use median as a robust estimate of "typical peanut area"
    typical = np.median(big_areas)

    # Accept blobs between [0.5 * typical, 1.5 * typical]
    # You can tighten/loosen these depending on your trays.
    min_area = 0.5 * typical
    max_area = 1.5 * typical

    peanut_labels = []
    for idx, area in enumerate(areas, start=1):  # +1 because we skipped background
        if min_area <= area <= max_area:
            peanut_labels.append(idx)

    count = len(peanut_labels)

    if debug:
        print(f"Total components (incl. background): {num_labels}")
        print(f"Typical peanut area ~ {typical:.1f}")
        print(f"Area range accepted: [{min_area:.1f}, {max_area:.1f}]")
        print(f"Detected peanuts: {count}")

    return count, peanut_labels, labels, stats


def draw_peanut_boxes(img_gray, peanut_labels, labels, stats):
    """
    Make an RGB copy of img_gray and draw bounding boxes + indices
    on components whose labels are in peanut_labels.
    """
    # Convert to color for drawing
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    for lbl in peanut_labels:
        x = stats[lbl, cv2.CC_STAT_LEFT]
        y = stats[lbl, cv2.CC_STAT_TOP]
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]

        # Draw rectangle and component id
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            vis,
            str(lbl),
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return vis


def main():

    img_path = "peanut count image.png"

    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        sys.exit(1)

    # Load as grayscale
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Failed to load image: {img_path}")
        sys.exit(1)

    # Count peanuts
    count, peanut_labels, labels, stats = count_peanuts(img_gray, debug=True)
    print(f"\nEstimated peanut count: {count}")

    # Optional: save debug visualization with boxes
    vis = draw_peanut_boxes(img_gray, peanut_labels, labels, stats)
    out_path = os.path.splitext(img_path)[0] + "_count_vis.png"
    cv2.imwrite(out_path, vis)
    print(f"Saved visualization to: {out_path}")


if __name__ == "__main__":
    main()
