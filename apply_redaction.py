import os
import shutil
import hashlib
import json
from typing import Optional
import logging

import fiftyone as fo
from fiftyone import ViewField as F

from redaction import mask_file_at, blur_file_at, anonymize_file_at

logger = logging.getLogger(__name__)

def hash_config(
        redaction_field,
        redaction_filter,
        redaction_type,
        redaction_method,
    ):
    return hashlib.sha256(json.dumps({
        "redaction_field": redaction_field,
        "redaction_filter": redaction_filter,
        "redaction_type": redaction_type,
        "redaction_method": redaction_method,
    }, sort_keys=True).encode()).hexdigest()


def make_redacted_filepath(
        filepath,
        redaction_config_hash,
    ):
    dirpath, filename = os.path.split(filepath)

    redacted_dir = os.path.join(dirpath, redaction_config_hash)
    if os.path.exists(redacted_dir) and not os.path.isdir(redacted_dir):
        raise FileExistsError(f"Cannot create redacted directory: {redacted_dir} exists and is not a directory")
    else:
        logger.info(f"Redactions will be saved to: {redacted_dir}")
        os.makedirs(redacted_dir, exist_ok=True)
    
    redacted_path = os.path.join(redacted_dir, filename)
    return redacted_path


def create_redactions(
        unredacted_dataset: fo.core.dataset.Dataset,
        redaction_field: str = "ground_truth",
        redaction_filter: Optional[dict] = {"supercategory": "person"},
        redaction_type: str = "bounding_box",  # options: "bounding_box", "mask"
        redaction_method: str = "mask",  # options: "mask", "blur", "anonymize"
        force_recreate: bool = False,
    ) -> None:
    """
    Given a FiftyOne view containing unredacted images,
    1. Makes redacted copies of the media at filepath --> saves to config_hash/filename path
    2. Creates samples with the redacted filepaths set as default
    3. Adds tags to the samples based on the redaction method
    """
    # TODO(neeraja): support mask-based redaction
    if redaction_type != "bounding_box":
        raise NotImplementedError("Currently only bounding box redaction is supported")
    
    for sample in unredacted_dataset:
        if ("blur" in sample.tags) or ("mask" in sample.tags) or ("anonymize" in sample.tags):
            logger.debug(f"Sample is a redacted sample with tags: {sample.tags}; skipping redaction...")
            continue
        
        redaction_config_hash = hash_config(
            redaction_field,
            redaction_filter,
            redaction_type,
            redaction_method,
        )
        redacted_path = make_redacted_filepath(
            sample.filepath,
            redaction_config_hash,
        )
        if sample.has_field("redacted_sample_ids") and sample.redacted_sample_ids and redaction_config_hash in sample.redacted_sample_ids:
            logger.debug(f"Sample has a redacted version with config hash = {redaction_config_hash}, id = {sample.redacted_sample_ids[redaction_config_hash]}; skipping redaction...")
            continue

        if os.path.exists(redacted_path) and not force_recreate:
            logger.debug(f"Redacted file already exists: {redacted_path}; skipping creation...")
        else:
            shutil.copy(sample.filepath, redacted_path)

            if redaction_method == "mask":
                mask_file_at(redacted_path, sample[redaction_field], redaction_filter)
            elif redaction_method == "blur":
                blur_file_at(redacted_path, sample[redaction_field], redaction_filter)
            elif redaction_method == "anonymize":
                anonymize_file_at(redacted_path, sample[redaction_field], redaction_filter)
            else:
                raise ValueError(f"Unknown redaction method: {redaction_method}")

        # duplicate the sample and set the redacted filepath
        redacted_sample = sample.copy()
        redacted_sample["filepath"] = redacted_path
        redaction_key_string = "_" + "_".join(f"{k}={v}" for k, v in redaction_filter.items()) if redaction_filter else ""
        redacted_sample.tags.append(f"redacted_{redaction_field}_{redaction_key_string}")
        redacted_sample.tags.append(redaction_type)
        redacted_sample.tags.append(redaction_method)
        logger.info(f"Adding redacted sample at filepath: {redacted_path}")

        # link the samples to each other
        sample["redacted_sample_ids"] = {
            redaction_config_hash: redacted_sample.id
        }
        sample.save()
        redacted_sample["original_sample_id"] = sample.id
        unredacted_dataset.add_sample(redacted_sample)
        redacted_sample.save()
    
    unredacted_dataset.save()
    return None
