import os
from typing import Optional
import logging

import cv2

import fiftyone as fo
from fiftyone import ViewField as F

from redaction import mask_file_at, blur_file_at, anonymize_file_at

logger = logging.getLogger(__name__)


def make_redacted_filepath(filepath):
    dirpath, filename = os.path.split(filepath)

    redacted_dir = os.path.join(dirpath, "redacted")
    if os.path.exists(redacted_dir) and not os.path.isdir(redacted_dir):
        raise FileExistsError(f"Cannot create redacted directory: {redacted_dir} exists and is not a directory")
    else:
        os.makedirs(redacted_dir, exist_ok=True)
    
    redacted_path = os.path.join(redacted_dir, filename)
    return redacted_path


def create_redactions(
        unredacted_view: fo.core.view.DatasetView,
        redaction_field: str = "ground_truth",
        redaction_filter: Optional[dict] = {"supercategory": "person"},
        redaction_type: str = "bounding_box",  # options: "bounding_box", "mask"
        redaction_method: str = "mask",  # options: "mask", "blur", "anonymize"
    ) -> fo.core.view.DatasetView:
    """
    Given a FiftyOne view containing unredacted images,
    1. Makes redacted copies of the media at filepath --> saves to redacted_filepath
    2. Returns a FiftyOne view with the redacted filepaths set as default
    """
    # TODO(neeraja): support mask-based redaction
    if redaction_type != "bounding_box":
        raise NotImplementedError("Currently only bounding box redaction is supported")
    
    # TODO(neeraja): maybe use clone_sample_field instead
    # https://github.com/voxel51/fiftyone/blob/bec3d00b274262f4b0416f7a0d8f0fe1f7e2ac99/plugins/operators/__init__.py#L396

    for sample in unredacted_view:
        redacted_path = make_redacted_filepath(sample.filepath)
        if os.path.exists(redacted_path):
            logger.debug(f"Redacted file already exists: {redacted_path}; overwriting...")
        os.system(f"cp '{sample.filepath}' '{redacted_path}'")

        if redaction_method == "mask":
            mask_file_at(redacted_path, sample[redaction_field], redaction_filter)
        elif redaction_method == "blur":
            blur_file_at(redacted_path, sample[redaction_field], redaction_filter)
        elif redaction_method == "anonymize":
            anonymize_file_at(redacted_path, sample[redaction_field], redaction_filter)
        else:
            raise ValueError(f"Unknown redaction method: {redaction_method}")
        sample["redacted_filepath"] = redacted_path
        sample.save()
    
    redacted_view = unredacted_view.set_field("filepath", F("redacted_filepath"))
    return redacted_view


def load_redactions(
        unredacted_view: fo.core.view.DatasetView,
    ) -> fo.core.view.DatasetView:
    """
    Given a FiftyOne view containing unredacted images,
    1. Makes redacted copies of the media at filepath --> saves to redacted_filepath
    2. Returns a FiftyOne view with the redacted filepaths set as default
    """
    for sample in unredacted_view:
        redacted_path = make_redacted_filepath(sample.filepath)
        if not os.path.exists(redacted_path):
            raise FileNotFoundError(f"Redacted file not found at: {redacted_path}")
        
        sample["redacted_filepath"] = redacted_path
        sample.save()
    
    redacted_view = unredacted_view.set_field("filepath", F("redacted_filepath"))
    return redacted_view
