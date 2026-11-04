"""Optional local frame overlays; no heavy dependency at module import time."""

from src.vision.video_io import require_opencv


def annotate_frame(frame, tracks, queue_track_ids, roi, entry_line, service_line,
                   queue_count, entries=0, services=0):
    """Return a BGR copy with anonymous queue geometry and person-only labels."""
    cv2 = require_opencv()
    import numpy as np

    image = frame.copy()
    height, width = image.shape[:2]
    pixel = lambda point: (round(point[0] * width), round(point[1] * height))
    polygon = np.array([pixel(point) for point in roi], dtype=np.int32)
    cv2.polylines(image, [polygon], True, (180, 165, 19), 2)  # teal BGR
    for line, label, color in (
        (entry_line, "ENTRY", (80, 170, 55)),
        (service_line, "SERVICE", (90, 105, 243)),
    ):
        start, end = map(pixel, line)
        cv2.line(image, start, end, color, 2)
        cv2.putText(image, label, (start[0], max(18, start[1] - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
    queued = set(queue_track_ids)
    for person in tracks:
        x1, y1, x2, y2 = (round(value) for value in person.bounding_box)
        in_queue = person.track_id in queued
        color = (93, 111, 243) if in_queue else (99, 59, 23)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"ID {person.track_id:02d}  {'QUEUE' if in_queue else 'OUTSIDE'}"
        cv2.putText(image, label, (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .46, color, 2)
        point = tuple(round(value) for value in person.representative_point)
        cv2.circle(image, point, 4, color, -1)
    cv2.rectangle(image, (0, 0), (width, 27), (23, 59, 99), -1)
    cv2.putText(image, f"QUEUE {queue_count}   ENTRY {entries}   SERVICE {services}",
                (9, 19), cv2.FONT_HERSHEY_SIMPLEX, .53, (255, 255, 255), 1)
    return image
