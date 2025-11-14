import os
import shutil
import hashlib
import json
from typing import Optional, Union
import logging

import fiftyone as fo
from fiftyone import ViewField as F

from redaction import redact_file_at

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


"""
Creates new samples
"""
def create_redaction_samples(
        # unredacted_dataset: fo.core.dataset.Dataset,
        unredacted_view: Union[fo.core.dataset.Dataset, fo.core.view.DatasetView],
        redaction_field: str = "ground_truth",
        redaction_filter: Optional[dict] = {"supercategory": "person"},
        redaction_type: str = "bounding_box",  # options: "bounding_box", "mask"
        redaction_method: str = "mask",  # options: "mask", "blur"
        force_recreate: bool = False,
    ) -> None:
    """
    Given a FiftyOne view containing unredacted images,
    1. Makes redacted copies of the media at filepath --> saves to config_hash/filename path
    2. Creates samples with the redacted filepaths set as default
    3. Adds tags to both the original and redacted samples based on the redaction method
    """
    redaction_config_hash = hash_config(
        redaction_field, redaction_filter, redaction_type, redaction_method
    )
    redaction_key_string = "_" + "_".join(f"{k}={v}" for k, v in redaction_filter.items()) if redaction_filter else ""
    redaction_tag = f"redacted_{redaction_field}_{redaction_key_string}_{redaction_type}_{redaction_method}"

    for sample in unredacted_view:
        if any("redacted_" in tag for tag in sample.tags):
            logger.debug(f"Sample is a redacted sample with tags: {sample.tags}; skipping redaction...")
            continue
        
        redacted_path = make_redacted_filepath(
            sample.filepath,
            redaction_config_hash,
        )
        if sample.has_field("redacted_sample_ids") and sample.redacted_sample_ids and redaction_config_hash in sample.redacted_sample_ids:
            logger.debug(f"Sample has a redacted version with config hash = {redaction_config_hash}, id = {sample.redacted_sample_ids[redaction_config_hash]}; skipping redaction...")
            continue

        if os.path.exists(redacted_path) and not force_recreate:
            logger.debug(f"Redacted file already exists: {redacted_path}; skipping creation...")
            continue
        else:
            shutil.copy(sample.filepath, redacted_path)
            redact_file_at(
                redacted_path,
                sample[redaction_field],
                redaction_filter=redaction_filter,
                redaction_type=redaction_type,
                redaction_method=redaction_method,
            )

        # duplicate the sample and set the redacted filepath
        redacted_sample = sample.copy()
        redacted_sample["redacted_sample_ids"] = {}
        redacted_sample["filepath"] = redacted_path
        redacted_sample.tags.append(redaction_tag)
        logger.info(f"Adding redacted sample at filepath: {redacted_path}")

        # link the samples to each other
        redacted_sample["original_sample_id"] = sample.id
        unredacted_view._dataset.add_sample(redacted_sample)
        sample["redacted_sample_ids"] = {
            redaction_config_hash: redacted_sample.id
        }
        sample.save()

    unredacted_view._dataset.save()
    return None


"""
Creates new media fields
"""
def create_redaction_fields(
        unredacted_view: Union[fo.core.dataset.Dataset, fo.core.view.DatasetView],
        redaction_field: str = "ground_truth",
        redaction_filter: Optional[dict] = {"supercategory": "person"},
        redaction_type: str = "bounding_box",  # options: "bounding_box", "mask"
        redaction_method: str = "mask",  # options: "mask", "blur"
        force_recreate: bool = False,
    ) -> None:
    """
    Given a FiftyOne view containing unredacted images,
    1. Makes redacted copies of the media at filepath --> saves to config_hash/filename path
    2. Adds a new media field with the redacted filepaths
    3. Adds a tag to the sample based on the redaction method
    """
    redaction_config_hash = hash_config(
        redaction_field, redaction_filter, redaction_type, redaction_method
    )
    redaction_key_string = "_" + "_".join(f"{k}={v}" for k, v in redaction_filter.items()) if redaction_filter else ""
    redaction_tag = f"redacted_{redaction_field}_{redaction_key_string}_{redaction_type}_{redaction_method}"

    for sample in unredacted_view:
        redacted_path = make_redacted_filepath(
            sample.filepath,
            redaction_config_hash,
        )
        if (redaction_tag in sample.tags) and os.path.exists(redacted_path) and not force_recreate:
            logger.debug(f"Sample has been redacted with tag: {redaction_tag}; Redacted file already exists: {redacted_path}; skipping redaction...")
            continue

        if os.path.exists(redacted_path) and not force_recreate:
            logger.debug(f"Redacted file already exists: {redacted_path}; skipping creation...")
            continue
        else:
            shutil.copy(sample.filepath, redacted_path)
            redact_file_at(
                redacted_path,
                sample[redaction_field],
                redaction_filter=redaction_filter,
                redaction_type=redaction_type,
                redaction_method=redaction_method,
            )
        
        # add the redacted filepath to the sample
        sample.tags.append(redaction_tag)
        sample[f"redacted_filepath_{redaction_tag}"] = redacted_path
        sample.save()

    logger.info(f"Adding redacted media field: redacted_filepath_{redaction_tag}")
    if f"redacted_filepath_{redaction_tag}" not in unredacted_view._dataset.app_config.media_fields:
       unredacted_view._dataset.app_config.media_fields.append(f"redacted_filepath_{redaction_tag}")

    unredacted_view._dataset.save()
    return None
