"""Image Redaction plugin.

| Copyright 2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`
"""

from dataclasses import field
import os
import IPython

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
from fiftyone.core.utils import add_sys_path

with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
    from apply_redaction import create_redaction_samples, create_redaction_fields


def _input_control_flow(ctx):
    inputs = types.Object()
    
    # ## redaction field
    active_fields = ctx.active_fields
    fields = ctx.view.get_field_schema(flat=True)

    #TODO(neeraja): remove the hardcoded "detections" attribute requirement
    detection_sources = [
        ff for ff in list(fields.keys())
        # if fields[ff].document_type == fo.core.labels.Detections
        if f"{ff}.detections" in fields
        and isinstance(fields[f"{ff}.detections"], fo.core.fields.ListField)
    ]
    
    # field_dropdown = types.Dropdown(choices=detection_sources)
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
        **enum_kwargs
    )
    chosen_redaction_field = ctx.params.get("redaction_field", field_dropdown.choices[0].value)

    ## redaction filter
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

    ## redaction type
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

    ## redaction method
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

    # ## redaction force recreate

    return types.Property(inputs)

class CreateRedactionSamples(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="create_redaction_samples", 
            label="Create Redactions as new samples Operator",
            description="Mask or blur sensitive detections in the dataset",
            icon="/assets/masks.svg",
        )

    def resolve_input(self, ctx):
        return _input_control_flow(ctx)

    def execute(self, ctx):
        #TODO(neeraja): why are we recreating?
        ctx.dataset.persistent = True
        create_redaction_samples(
            ctx.dataset,
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
        )

    def resolve_input(self, ctx):
        return _input_control_flow(ctx)

    def execute(self, ctx):
        #TODO(neeraja): why are we recreating?
        ctx.dataset.persistent = True
        create_redaction_fields(
            ctx.dataset,
            redaction_field=ctx.params.get("redaction_field"),
            redaction_filter=ctx.params.get("redaction_filter"),
            redaction_type=ctx.params.get("redaction_type_choices", "bounding_box"),
            redaction_method=ctx.params.get("redaction_method_choices", "mask"),
        )
        return {}


def register(p):
    p.register(CreateRedactionSamples)
    p.register(CreateRedactionFields)
