from typing import Optional, Iterable
import logging
import numpy as np
import cv2
from typing import Tuple, List
import fiftyone as fo

logger = logging.getLogger(__name__)


GAUSSIAN_BLUR_KERNEL_SIZE = lambda w: (w // 10) * 2 + 1  # ensure ksize is odd


def filter_detections_by_dict(
        detections_object: fo.core.labels.Detections,
        redaction_filter: Optional[dict] = None,
    ) -> Iterable:
    """
    Filters the detections in the given Detections object based on the redaction_filter.

    Args:
        detections_object: fo.core.labels.Detections attribute of the sample
        redaction_filter: dict specifying attribute-value pairs to filter detections
    Returns:
        A new iterable containing only the filtered detections
    """
    #TODO(neeraja): remove the hardcoded "detections" attribute
    detections_iterable = detections_object.detections if detections_object else []
    detections_iterable = detections_iterable or []
    if not redaction_filter:
        yield from detections_iterable
        return

    for detection in detections_iterable:
        if all(getattr(detection, k, None) == v for k, v in redaction_filter.items()):
            yield detection


def filter_detections(
        detections_object: fo.core.labels.Detections,
        redaction_labels: List[str],
    ) -> Iterable:
    """
    Filters the detections in the given Detections object based on the redaction_filter.

    Args:
        detections_object: fo.core.labels.Detections attribute of the sample
        redaction_labels: List[str] of the labels to filter detections
    Returns:
        A new iterable containing only the filtered detections
    """
    #TODO(neeraja): remove the hardcoded "detections" attribute
    detections_iterable = detections_object.detections if detections_object else []
    detections_iterable = detections_iterable or []

    for detection in detections_iterable:
        #TODO(neeraja): decide whether to keep the hardcoded "label" key
        if detection["label"] in redaction_labels:
            yield detection


def fit_mask_to_bbox(mask: np.ndarray, bbox_size: Tuple[int, int]) -> np.ndarray:
    """
    Pads or crops the mask to the bounding box size.
    Args:
        mask: np.ndarray of shape (mask_height, mask_width)
        bbox_size: Tuple[int, int] of the bounding box size (height, width)
    Returns:
        np.ndarray of shape (height, width)
    """
    return np.pad(mask, [(0, max(0, bbox_size[0] - mask.shape[0])), (0, max(0, bbox_size[1] - mask.shape[1]))])[:bbox_size[0], :bbox_size[1]]


def get_corners_from_bbox(bbox: Tuple[float, float, float, float], image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Returns the corners of the bounding box in image coordinates.

    Args:
        bbox: Tuple[float, float, float, float] of the bounding box (top-left-x, top-left-y, width, height)
        image_shape: Tuple[int, int] of the image shape (height, width)
    
    Returns:
        Tuple[int, int, int, int] of the corners of the bounding box
        in image coordinates (left, top, right, bottom) i.e. (x1, y1, x2, y2)
    """
    x1 = int(bbox[0] * image_shape[1])
    y1 = int(bbox[1] * image_shape[0])
    x2 = int((bbox[0] + bbox[2]) * image_shape[1])
    y2 = int((bbox[1] + bbox[3]) * image_shape[0])
    return (x1, y1, x2, y2)


def redact_file_at(
        filepath: str,
        detections_object: fo.core.labels.Detections,
        # redaction_filter: Optional[dict] = None,
        redaction_labels: List[str] = ["person"],
        redaction_type: str = "bounding_box",  # options: "bounding_box", "segmentation_mask"
        redaction_method: str = "mask",  # options: "mask", "blur"
    ) -> None:
    """
    Blurs or masksout the regions specified by the detections in the image at filepath.

    Args:
        filepath: media file to edit
        detections_object: fo.core.labels.Detections attribute of the sample
    """
    image = cv2.imread(filepath)
    if redaction_method == "blur":
        try:
            max_box_size = max([
                max(detection.bounding_box[2] * image.shape[1], detection.bounding_box[3] * image.shape[0])
                for detection in filter_detections(detections_object, redaction_labels)
            ])
        except ValueError as e:
            # no detections found for redaction filter
            return
        ksize = int(max(GAUSSIAN_BLUR_KERNEL_SIZE(max_box_size), 1))
        redacted_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif redaction_method == "mask":
        redacted_image = np.zeros_like(image)
    else:
        raise ValueError(f"Unimplemented redaction method: {redaction_method}")

    for detection in filter_detections(detections_object, redaction_labels):
        x1, y1, x2, y2 = get_corners_from_bbox(detection.bounding_box, image.shape)
        mask = detection.get_mask()
        if redaction_type == "segmentation_mask" and mask is not None:
            mask = fit_mask_to_bbox(mask, (y2-y1, x2-x1))
            image[y1:y2, x1:x2][mask] = redacted_image[y1:y2, x1:x2][mask]
        elif redaction_type == "bounding_box":
            image[y1:y2, x1:x2] = redacted_image[y1:y2, x1:x2]
        elif redaction_type == "segmentation_mask" and mask is None:
            logger.warning(f"No mask found for detection: {detection}")
            continue
        else:
            raise ValueError(f"Unknown redaction type: {redaction_type}")
    
    cv2.imwrite(filepath, image)
    return
