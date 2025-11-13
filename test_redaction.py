### Imports

import time

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

from apply_redaction import create_redactions


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

create_redactions(
    persons,
    redaction_field="sam",
    redaction_filter={"label": "person"},
    redaction_type="segmentation_mask",
    redaction_method="mask",
)

### Launch App

session = fo.launch_app(persons)


### Zero Shot Instance Segmentation

# zsis = foo.get_operator("@jacobmarks/zero_shot_prediction/zero_shot_instance_segment")
# await zsis(twenty_samples, labels=["human_face"], model_name="SAM", label_field="sam")
# await zsis(twenty_samples, labels=["person"], model_name="SAM", label_field="sam")

