"""Image Redaction plugin.

| Copyright 2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`
"""

import os
import re

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.utils.transformers as fout
from fiftyone.core.utils import add_sys_path

with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
    from apply_redaction import create_redaction_samples, create_redaction_fields


MODEL_CONFIGS = {
    # ("person", "bounding_box"),
    # ("person", "segmentation_mask"),
    "redaction_labels = [face] | redaction_type = bounding_box": {
        "detection_model_name": "EsraaFouad/detr_fine_tune_face_detection_final",
        "segmentation_model_name": None,
        "field_name": "detr-ft-face",
        "confidence_thresh": 0.5,
    },
    "redaction_labels = [face] | redaction_type = segmentation_mask": {
        "detection_model_name": "EsraaFouad/detr_fine_tune_face_detection_final",
        "segmentation_model_name": "segment-anything-vitb-torch",
        "field_name": "detr-ft-face",
        "confidence_thresh": 0.5,
    },
    # ("human_face", "bounding_box"),
    # ("human_face", "segmentation_mask"),
    # ("license_plate", "bounding_box"),
    # ("license_plate", "segmentation_mask"),
}


def _get_device_name():
    import torch
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 0:
            return "cuda:0"
        else:
            return "cpu"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def _get_detection_model(model_name):
    # TODO(neeraja): ensure this works for other models too
    from transformers import AutoModelForObjectDetection
    detr_model = AutoModelForObjectDetection.from_pretrained(model_name)
    detr_model.to(_get_device_name())
    return fout.convert_transformers_model(detr_model, task="object-detection")


def _get_segmentation_model(model_name):
    import torch
    fo_model = foz.load_zoo_model("segment-anything-vitb-torch", device=_get_device_name())
    device_name = _get_device_name()
    fo_model.config.device = device_name
    fo_model._device = torch.device(device_name)
    fo_model._using_gpu = True  # MPS is considered GPU-like
    fo_model._model.to(device_name)
    return fo_model


def _input_control_flow(ctx):
    inputs = types.Object()
    
    # Options: Use existing detections or apply model to make new detections
    detection_mode_choices = types.RadioGroup()
    detection_mode_choices.add_choice("use_existing", label="Use existing sensitive content detections")
    detection_mode_choices.add_choice("apply_model", label="Apply a model for sensitive content detections")
    inputs.enum(
        "detection_mode",
        detection_mode_choices.values(),
        label="Sensitive Content Detection Source",
        view=detection_mode_choices,
        required=True,
    )
    
    # Get the selected mode - only show conditional fields if user has made a selection
    detection_mode = ctx.params.get("detection_mode")
    
    # Only show conditional fields after user makes a selection
    if detection_mode == "use_existing":
        # Redaction field
        fields = ctx.view.get_field_schema(flat=True)
        #TODO(neeraja): remove the hardcoded "detections" attribute requirement
        detection_sources = [
            ff for ff in list(fields.keys())
            if f"{ff}.detections" in fields
            and isinstance(fields[f"{ff}.detections"], fo.core.fields.ListField)
        ]
        
        field_dropdown = types.Dropdown()
        for source in detection_sources:
            field_dropdown.add_choice(source, label=source)
        inputs.enum(
            "redaction_field",
            field_dropdown.values(),
            default="ground_truth",
            label="Field in which redaction label detections are stored",
            view=field_dropdown,
            multiple=False,
            required=True,
        )

        ## Note: getting rid of the dict-style filter in favor of "label": <redaction_labels>
        # # Redaction filter
        # autocomplete_view_keys = types.AutocompleteView()
        # for choice_key in ["category", "supercategory", "classification", "label"]:
        #     autocomplete_view_keys.add_choice(choice_key, label=choice_key)
        # autocomplete_view_values = types.AutocompleteView()
        # for choice_value in ["face", "person", "number", "license plate"]:
        #     autocomplete_view_values.add_choice(choice_value, label=choice_value)
        # inputs.map(
        #     "redaction_filter",
        #     key_type=types.String(),
        #     key=autocomplete_view_keys,
        #     value_type=types.String(),
        #     value=autocomplete_view_values,
        #     label="Redaction Filter (key: value list)",
        #     description="Filter the redactions by a specific field",
        # )
        
        # Redaction Label
        # TODO(neeraja): populate a dropdown based on existing labels in the dataset
        inputs.str(
            "redaction_labels",
            default="person",
            label="Labels to redact (comma-separated)",
            required=True,
        )

        # Redaction type
        radio_choices = types.RadioGroup()
        radio_choices.add_choice("bounding_box", label="Bounding Box")
        radio_choices.add_choice("segmentation_mask", label="Segmentation Mask")
        inputs.enum(
            "redaction_type_choices",
            radio_choices.values(),
            default=radio_choices.choices[0].value,
            label="Redaction Area Type",
            view=radio_choices,
        )

        # Redaction method
        radio_choices = types.RadioGroup()
        radio_choices.add_choice("mask", label="Mask")
        radio_choices.add_choice("blur", label="Blur")
        radio_choices.add_choice("anonymize", label="Anonymize")
        inputs.enum(
            "redaction_method_choices",
            radio_choices.values(),
            default=radio_choices.choices[0].value,
            label="Redaction Method to Apply",
            view=radio_choices,
        )

    elif detection_mode == "apply_model":
        model_config_dropdown = types.Dropdown()
        for model_config_key in MODEL_CONFIGS.keys():
            model_config_dropdown.add_choice(model_config_key, label=model_config_key)
        
        inputs.enum(
            "model_config_key_choices",
            model_config_dropdown.values(),
            # default=model_config_dropdown.choices[0].value if len(model_config_dropdown.choices) > 0 else None,
            label="Sensitive Content Detection Model Configuration",
            view=model_config_dropdown,
            required=True,
        )
        model_config_key = ctx.params.get("model_config_key_choices")

        # Redaction method
        radio_choices = types.RadioGroup()
        radio_choices.add_choice("mask", label="Mask")
        radio_choices.add_choice("blur", label="Blur")
        radio_choices.add_choice("anonymize", label="Anonymize")
        inputs.enum(
            "redaction_method_choices",
            radio_choices.values(),
            default=radio_choices.choices[0].value,
            label="Redaction Method to Apply",
            view=radio_choices,
        )

    return types.Property(inputs)


def _model_config_resolution_flow(ctx):
    model_config_key = ctx.params.get("model_config_key_choices")
    if model_config_key is None:
        raise ValueError("Model configuration key must be selected")
    else:
        redaction_labels, redaction_type = model_config_key.split("|")
        redaction_labels = redaction_labels.split("=")[1].strip().strip("[").strip("]")
        redaction_type = redaction_type.split("=")[1].strip()
        ctx.params["redaction_labels"] = redaction_labels
        ctx.params["redaction_type_choices"] = redaction_type
        model_config = MODEL_CONFIGS[model_config_key]
        ctx.params["detection_model_name"] = model_config["detection_model_name"]
        ctx.params["confidence_thresh"] = model_config["confidence_thresh"]
        ctx.params["redaction_field"] = model_config["field_name"]
    
    return ctx


def _model_application_flow(ctx, view):
    view.apply_model(
        model=_get_detection_model(ctx.params.get("detection_model_name")),
        label_field=ctx.params.get("redaction_field"),
        confidence_thresh=ctx.params.get("confidence_thresh"),
    )
    if ctx.params.get("redaction_type_choices") == "segmentation_mask":
        view.apply_model(
            model=_get_segmentation_model(ctx.params.get("segmentation_model_name")),
            label_field=ctx.params.get("redaction_field"),
            prompt_field=ctx.params.get("redaction_field"),
        )
    return view

class CreateRedactionSamples(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="create_redaction_samples", 
            label="Create Redactions as new samples Operator",
            description="Mask or blur sensitive detections in the dataset",
            icon="/assets/masks.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        return _input_control_flow(ctx)

    def execute(self, ctx):
        ctx.dataset.persistent = True
        view = ctx.target_view()
        detection_mode = ctx.params.get("detection_mode")
        
        if detection_mode is None:
            raise ValueError("Detection mode must be selected")
        
        if detection_mode == "apply_model":
            ctx =_model_config_resolution_flow(ctx)
            view = _model_application_flow(ctx, view)

        create_redaction_samples(
            view,
            redaction_field=ctx.params.get("redaction_field"),
            redaction_labels=[ll.strip() for ll in ctx.params.get("redaction_labels").split(",")],
            redaction_type=ctx.params.get("redaction_type_choices", "bounding_box"),
            redaction_method=ctx.params.get("redaction_method_choices", "mask"),
        )
        return {}

class CreateRedactionFields(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="create_redaction_fields", 
            label="Create Redactions as new media fields Operator",
            description="Mask or blur sensitive detections in the dataset",
            icon="/assets/masks.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        return _input_control_flow(ctx)

    def execute(self, ctx):
        ctx.dataset.persistent = True
        view = ctx.target_view()
        detection_mode = ctx.params.get("detection_mode")
        
        if detection_mode is None:
            raise ValueError("Detection mode must be selected")
        
        if detection_mode == "apply_model":
            ctx =_model_config_resolution_flow(ctx)
            view = _model_application_flow(ctx, view)
        
        create_redaction_fields(
            view,
            redaction_field=ctx.params.get("redaction_field"),
            redaction_labels=[ll.strip() for ll in ctx.params.get("redaction_labels").split(",")],
            redaction_type=ctx.params.get("redaction_type_choices", "bounding_box"),
            redaction_method=ctx.params.get("redaction_method_choices", "mask"),
        )
        return {}


def register(p):
    p.register(CreateRedactionSamples)
    p.register(CreateRedactionFields)
