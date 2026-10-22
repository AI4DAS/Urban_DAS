"""Associate YOLO patch detections into vehicle traces."""

from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def _metric(test_segment, tracked_segment):
    test_vector = np.subtract(test_segment[1], test_segment[0])
    tracked_vector = np.subtract(tracked_segment[1], tracked_segment[0])
    denominator = np.linalg.norm(test_vector) * np.linalg.norm(tracked_vector)
    cosine = float(test_vector.dot(tracked_vector) / denominator) if denominator else -1.0
    distance = min(
        np.linalg.norm(np.subtract(tracked_segment[1], test_segment[0])),
        np.linalg.norm(np.subtract(tracked_segment[0], test_segment[1])),
    )
    return cosine - 0.2 * distance + float(test_segment[2] == tracked_segment[2])


def _match(detections, trackers, threshold):
    if not trackers or not detections:
        return [], list(range(len(detections)))

    scores = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for detection_index, detection in enumerate(detections):
        for tracker_index, tracker in enumerate(trackers):
            scores[detection_index, tracker_index] = _metric(detection, tracker[-1])

    rows, columns = linear_sum_assignment(-scores)
    matches = []
    matched_detections = set()
    for detection_index, tracker_index in zip(rows, columns):
        if scores[detection_index, tracker_index] >= threshold:
            matches.append((detection_index, tracker_index))
            matched_detections.add(int(detection_index))
    unmatched = [index for index in range(len(detections)) if index not in matched_detections]
    return matches, unmatched


def _box_to_segment(label_file, patch_image_size):
    segments = []
    for row in Path(label_file).read_text(encoding="utf-8").splitlines():
        if not row.strip():
            continue
        label, x, y, width, height = map(float, row.split()[:5])
        x1 = round((x - 0.5 * width) * patch_image_size - 0.5)
        x2 = round((x + 0.5 * width) * patch_image_size - 0.5)
        y1 = round((y - 0.5 * height) * patch_image_size - 0.5)
        y2 = round((y + 0.5 * height) * patch_image_size - 0.5)
        if int(label) == 0:
            segments.append([[x2, y1], [x1, y2], 0])
        elif int(label) == 1:
            segments.append([[x1, y1], [x2, y2], 1])
    return segments


def _scan_direction(
    image_name,
    labels_dir,
    rows,
    columns,
    direction,
    patch_size,
    padding,
    threshold,
):
    trackers = []
    column_order = range(columns) if direction == 1 else range(columns - 1, -1, -1)
    class_id = 1 if direction == 1 else 0
    detection_count = 0
    patch_image_size = patch_size + 2 * padding

    for row in range(rows):
        for column in column_order:
            patch_id = row * columns + column
            label_file = labels_dir / f"{image_name}_{patch_id:05d}.txt"
            detections = []
            if label_file.exists():
                for point1, point2, label in _box_to_segment(label_file, patch_image_size):
                    if label != class_id:
                        continue
                    offset = np.array([column * patch_size - padding, row * patch_size - padding])
                    detections.append(
                        [(np.asarray(point1) + offset).tolist(), (np.asarray(point2) + offset).tolist(), label]
                    )
                    detection_count += 1

            matches, unmatched = _match(detections, trackers, threshold)
            for detection_index, tracker_index in matches:
                trackers[tracker_index].append(detections[detection_index])
            for detection_index in unmatched:
                trackers.append([detections[detection_index]])

    return trackers, detection_count


def associate_image(
    image_name,
    labels_dir,
    image_height,
    image_width,
    *,
    patch_size=64,
    padding=16,
    threshold=0.1,
    minimum_segments=1,
):
    """Return the legacy-compatible association payload for one image."""
    labels_dir = Path(labels_dir)
    rows = image_height // patch_size
    columns = image_width // patch_size
    negative, negative_count = _scan_direction(
        image_name, labels_dir, rows, columns, 1, patch_size, padding, threshold
    )
    positive, positive_count = _scan_direction(
        image_name, labels_dir, rows, columns, -1, patch_size, padding, threshold
    )
    trackers = [trace for trace in positive + negative if len(trace) >= minimum_segments]
    total = positive_count + negative_count
    return {
        "img": [image_height, image_width],
        "crop img": [rows * patch_size, columns * patch_size],
        "associate pixel": trackers,
        "associate performance": f"from total #{total} to {minimum_segments}#{len(trackers)}",
    }
