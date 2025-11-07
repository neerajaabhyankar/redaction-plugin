from typing import Optional, Iterable
import cv2
import fiftyone as fo


def filter_detections(
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
    detections_iterable = detections_object.detections if detections_object else []
    detections_iterable = detections_iterable or []
    if not redaction_filter:
        yield from detections_iterable
        return

    for detection in detections_iterable:
        if all(getattr(detection, k, None) == v for k, v in redaction_filter.items()):
            yield detection


def mask_file_at(
        filepath: str,
        detections_object: fo.core.labels.Detections,
        redaction_filter: Optional[dict] = None,
    ) -> None:
    """
    Blacks out the regions specified by the detections in the image at filepath.

    Args:
        filepath: media file to edit
        detections_object: fo.core.labels.Detections attribute of the sample
    """
    image = cv2.imread(filepath)
    for detection in filter_detections(detections_object, redaction_filter):
        bbox = detection.bounding_box  # [top-left-x, top-left-y, width, height]
        x1 = int(bbox[0] * image.shape[1])
        y1 = int(bbox[1] * image.shape[0])
        x2 = int((bbox[0] + bbox[2]) * image.shape[1])
        y2 = int((bbox[1] + bbox[3]) * image.shape[0])
        image[y1:y2, x1:x2] = 0  # black rectangle
    
    cv2.imwrite(filepath, image)
    return


def blur_file_at(
        filepath: str,
        detections_object: fo.core.labels.Detections,
        redaction_filter: Optional[dict] = None,
    ) -> None:
    """
    Blurs out the regions specified by the detections in the image at filepath.

    Args:
        filepath: media file to edit
        detections_object: fo.core.labels.Detections attribute of the sample
    """
    image = cv2.imread(filepath)
    # TODO(neeraja): adjust ksize based on bbox size
    ksize = image.shape[0] // 20 * 2 + 1  # ensure ksize is odd
    blurred_image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    for detection in filter_detections(detections_object, redaction_filter):
        bbox = detection.bounding_box  # [top-left-x, top-left-y, width, height]
        x1 = int(bbox[0] * image.shape[1])
        y1 = int(bbox[1] * image.shape[0])
        x2 = int((bbox[0] + bbox[2]) * image.shape[1])
        y2 = int((bbox[1] + bbox[3]) * image.shape[0])
        image[y1:y2, x1:x2] = blurred_image[y1:y2, x1:x2]  # blur rectangle
    
    cv2.imwrite(filepath, image)
    return


def anonymize_file_at(
        filepath: str,
        detections_object: fo.core.labels.Detections,
        redaction_filter: Optional[dict] = None,
    ) -> None:
    """
    Replaces the regions specified by the detections in the image at filepath.

    Args:
        filepath: media file to edit
        detections_object: fo.core.labels.Detections attribute of the sample
    """
    raise NotImplementedError("Anonymization not yet implemented")
