### Imports

import time

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

from apply_redaction import create_redaction_samples, create_redaction_fields


# ### Load COCO Dataset

# dataset = foz.load_zoo_dataset("coco-2017", split="validation")

# dataset.persistent = True
# create_redactions(
#     dataset,
#     redaction_field="ground_truth",
#     redaction_filter={"supercategory": "person"},
#     redaction_type="bounding_box",
#     redaction_method="mask",
# )

# ### Test without GUI

# import fiftyone.operators as foo

# ctx = {
#     "selected": [
#         "69141a60322a219721f89461",
#         "69141a60322a219721f89485",
#     ],
#     "active_fields": ["ground_truth"],
#     "dataset": dataset,
#     "view": dataset[:10],
# }

# # red = foo.execute_operator("@neerajaabhyankar-redaction-plugin-local/create_redactions", ctx)
# red = foo.execute_operator("@neerajaabhyankar/redaction-plugin/create_redactions", ctx)

# ### Launch App

# # only show samples with persons
# persons = dataset.filter_labels("ground_truth", F("supercategory") == "person")
# session = fo.launch_app(persons)

# input("Press Enter to close the app and exit...")
# session.close()
# time.sleep(5)

### Load Quickstart Dataset

dataset = fo.load_dataset("quickstart")

persons = dataset.filter_labels("sam", F("label") == "person")

redacted_persons = create_redaction_fields(
    persons,
    redaction_field="sam",
    redaction_filter={"label": "person"},
    redaction_type="segmentation_mask",
    redaction_method="blur",
)
# redacted_persons.app_config.grid_media_field = f"redacted_filepath_sam_label=person_segmentation_mask_blur"

### Launch App

session = fo.launch_app(redacted_persons)


### Zero Shot Instance Segmentation

# zsis = foo.get_operator("@jacobmarks/zero_shot_prediction/zero_shot_instance_segment")
# await zsis(twenty_samples, labels=["human_face"], model_name="SAM", label_field="sam")
# await zsis(twenty_samples, labels=["person"], model_name="SAM", label_field="sam")

# ### Test without GUI


import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import fiftyone.operators as foo

dataset = foz.load_zoo_dataset("quickstart")

ctx = {
    "active_fields": ["ground_truth"],
    "dataset": dataset,
    # "view": dataset[:10],
    # "view": dataset.select("6916e49628d06b5c1fc39f8a"),
    "params": {
        "redaction_field": "ground_truth",
        "redaction_filter": {"label": "person"},
        "redaction_type_choices": "bounding_box",
        "redaction_method_choices": "blur",
    },
}

# # red = foo.execute_operator("@neerajaabhyankar-redaction-plugin-local/create_redactions", ctx)
red = foo.execute_operator("@neerajaabhyankar/redaction-plugin/create_redaction_fields", ctx)

# # logging

# import logging
# LOGPATH = "/tmp/fologs/debug2.log"
# def setup_file_logger():
#     logger = logging.getLogger("redaction_logger")
#     if not logger.handlers:
#         h = logging.FileHandler(LOGPATH)
#         fmt = logging.Formatter("%(asctime)s %(process)d %(threadName)s %(levelname)s %(message)s")
#         h.setFormatter(fmt)
#         logger.addHandler(h)
#         logger.setLevel(logging.DEBUG)
#     return logger
# logger = setup_file_logger()