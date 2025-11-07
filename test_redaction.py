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

# redacted_dataset = create_redactions(
#     dataset,
#     redaction_field="ground_truth",
#     redaction_filter={"supercategory": "person"},
#     redaction_type="bounding_box",
#     redaction_method="blur",
# )

redacted_dataset = load_redactions(dataset)

### Launch App

session = fo.launch_app(redacted_dataset)
# session = fo.launch_app(persons)  # for the original images

input("Press Enter to close the app and exit...")
session.close()
time.sleep(5)