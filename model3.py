import os
import sys

sys.path.insert(0, "/Users/neeraja/fiftyone")
os.environ["PYTHONPATH"] = "/Users/neeraja/fiftyone"

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

import fiftyone.utils.transformers as fout
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def _get_segmentation_model(architecture):
    zoo_model_name = (
        "segment-anything-" + architecture.lower().replace("-", "") + "-torch"
    )
    return foz.load_zoo_model(zoo_model_name)


if __name__ == "__main__":
    dataset = fo.load_dataset("quickstart")
    # two_samples = dataset.take(2)
    # twenty_samples = dataset.take(20)

    # Load the processor and model from Hugging Face
    # The processor will be automatically used by the FiftyOne model wrapper
    detr_processor = AutoImageProcessor.from_pretrained("EsraaFouad/detr_fine_tune_face_detection_final")
    detr_model = AutoModelForObjectDetection.from_pretrained("EsraaFouad/detr_fine_tune_face_detection_final")
    label_field = "face-det"

    # Convert the transformers model to a FiftyOne model
    # The processor is automatically loaded from the model's config, but you can
    # verify it's available after conversion via fo_model.transforms.processor
    fo_model = fout.convert_transformers_model(detr_model, task="object-detection")

    # Apply the detection model to samples
    dataset.apply_model(fo_model, label_field=label_field, confidence_thresh=0.8)

    # # Apply the segmentation model to samples
    # seg_model = _get_segmentation_model(seg_architecture)
    # dataset.apply_model(
    #     seg_model, label_field=label_field, prompt_field=label_field
    # )

    session = fo.launch_app(dataset)
    input("Press Enter to close the app and exit...")
    session.close()
    print("Model applied successfully!")
