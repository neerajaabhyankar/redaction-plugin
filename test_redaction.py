### Imports

import time
import os

import cv2

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

from apply_redaction import make_redacted_filepath, create_redactions, load_redactions


### Load Dataset

dataset = foz.load_zoo_dataset("coco-2017", split="validation")

### Apply Model to get labels

# from model2 import apply_model_adhoc

# samples = dataset.take(10)
# predicted_samples = apply_model_adhoc(samples)

# session = fo.launch_app(predicted_samples)

# redacted_dataset = create_redactions(
#     dataset,
#     redaction_field="predictions-yakhyo-tinyface",
#     redaction_filter={"label": "face"},
#     redaction_type="bounding_box",
#     redaction_method="blur",
# )

# redacted_dataset = create_redactions(
#     dataset,
#     redaction_field="ground_truth",
#     redaction_filter={"supercategory": "person"},
#     redaction_type="bounding_box",
#     redaction_method="blur",
# )

### Load Previous "Person" Redactions

redacted_dataset = load_redactions(dataset)

### Launch App

session = fo.launch_app(redacted_dataset.filter_labels("ground_truth", F("supercategory") == "person"))
# session = fo.launch_app(redacted_dataset)
# session = fo.launch_app(dataset)  # for the original images

input("Press Enter to close the app and exit...")
session.close()
time.sleep(5)