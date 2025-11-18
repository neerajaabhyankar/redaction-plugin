"""Image Redaction plugin.

| Copyright 2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`
"""

import os

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
from fiftyone.core.utils import add_sys_path

with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
    from apply_redaction import create_redaction_samples, create_redaction_fields


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

        enum_kwargs = {
            "label": "Redaction Field",
            "view": field_dropdown,
            "multiple": False,
        }
        if len(field_dropdown.choices) > 0:
            enum_kwargs["default"] = field_dropdown.choices[0].value
        inputs.enum(
            "redaction_field",
            field_dropdown.values(),
            required=True,
            **enum_kwargs
        )

        # Redaction filter
        autocomplete_view_keys = types.AutocompleteView()
        for choice_key in ["category", "supercategory", "classification", "label"]:
            autocomplete_view_keys.add_choice(choice_key, label=choice_key)
        autocomplete_view_values = types.AutocompleteView()
        for choice_value in ["face", "person", "number", "license plate"]:
            autocomplete_view_values.add_choice(choice_value, label=choice_value)
        inputs.map(
            "redaction_filter",
            key_type=types.String(),
            key=autocomplete_view_keys,
            value_type=types.String(),
            value=autocomplete_view_values,
            label="Redaction Filter (key: value list)",
            description="Filter the redactions by a specific field",
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
        # Model config dropdown
        model_config_dropdown = types.Dropdown()
        
        MODEL_CONFIGS = [
            ("human_face", "bounding_box"),
            ("human_face", "segmentation_mask"),
            ("face", "bounding_box"),
            ("face", "segmentation_mask"),
            ("license_plate", "bounding_box"),
            ("license_plate", "segmentation_mask"),
        ]
        for model_config in MODEL_CONFIGS:
            model_config_dropdown.add_choice(model_config, label=" ".join(model_config))
        
        inputs.enum(
            "model_config_choices",
            model_config_dropdown.values(),
            default=model_config_dropdown.choices[0].value if len(model_config_dropdown.choices) > 0 else None,
            label="Sensitive Content Detection Model Configuration",
            view=model_config_dropdown,
            required=True,
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

        model_config = ctx.params.get("model_config_choices")

        """
        1. determine the redaction field
        2. determine the redaction filter
        """

        ctx.params["redaction_type_choices"] = model_config[1]
        


    return types.Property(inputs)

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
        detection_mode = ctx.params.get("detection_mode")
        
        if detection_mode is None:
            raise ValueError("Detection mode must be selected")
        
        if detection_mode == "apply_model":
            # TODO: Implement model-based detection logic
            model_config = ctx.params.get("model_config")
            raise NotImplementedError(
                f"Model-based detection not yet implemented. "
                f"Selected config: {model_config}"
            )
        
        create_redaction_samples(
            ctx.view,
            redaction_field=ctx.params.get("redaction_field"),
            redaction_filter=ctx.params.get("redaction_filter"),
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
        detection_mode = ctx.params.get("detection_mode")
        
        if detection_mode is None:
            raise ValueError("Detection mode must be selected")
        
        if detection_mode == "apply_model":
            # TODO: Implement model-based detection logic
            model_config = ctx.params.get("model_config")
            raise NotImplementedError(
                f"Model-based detection not yet implemented. "
                f"Selected config: {model_config}"
            )
        
        create_redaction_fields(
            ctx.view,
            redaction_field=ctx.params.get("redaction_field"),
            redaction_filter=ctx.params.get("redaction_filter"),
            redaction_type=ctx.params.get("redaction_type_choices", "bounding_box"),
            redaction_method=ctx.params.get("redaction_method_choices", "mask"),
        )
        return {}


def register(p):
    p.register(CreateRedactionSamples)
    p.register(CreateRedactionFields)
