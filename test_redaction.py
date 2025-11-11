### Imports

import time
import os

import cv2

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

from apply_redaction import create_redactions


### Load Dataset

dataset = foz.load_zoo_dataset("coco-2017", split="validation")

create_redactions(
    dataset,
    redaction_field="ground_truth",
    redaction_filter={"supercategory": "person"},
    redaction_type="bounding_box",
    redaction_method="blur",
)

### Launch App

# only show samples with persons
persons = dataset.filter_labels("ground_truth", F("supercategory") == "person")
session = fo.launch_app(persons)

input("Press Enter to close the app and exit...")
session.close()
time.sleep(5)