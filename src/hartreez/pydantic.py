"""Pydantic v2 integration for unit-annotated model input.

Import this module explicitly to use :class:`UnitAwareModel`; the core
``hartreez`` namespace remains independent of Pydantic.
"""

from __future__ import annotations

import types
from collections.abc import Mapping
from typing import Annotated, Any, ClassVar, Union, cast, get_args, get_origin

from pydantic import BaseModel, ValidationInfo, model_validator
from pydantic.fields import FieldInfo

from hartreez.errors import MappingConversionError, ReferenceUnitsError
from hartreez.mapping import normalize_mapping_level, parse_reference_units
from hartreez.metadata import UnitDimension
from hartreez.systems import UnitSystem


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _field_dimension(name: str, model_field: FieldInfo) -> UnitDimension | None:
    candidates = [m for m in model_field.metadata if isinstance(m, UnitDimension)]

    # Pydantic only lifts Annotated metadata from the top of an annotation, so
    # an optional field such as ``Annotated[float, UnitDimension("K")] | None``
    # keeps its dimension inside the union members.
    if get_origin(model_field.annotation) in (Union, types.UnionType):
        for member in get_args(model_field.annotation):
            if get_origin(member) is Annotated:
                candidates.extend(
                    m for m in get_args(member)[1:] if isinstance(m, UnitDimension)
                )

    if not candidates:
        return None
    first = candidates[0]
    for other in candidates[1:]:
        if other.dimensions != first.dimensions:
            raise TypeError(
                f"field {name!r} declares conflicting UnitDimension metadata: "
                f"{first.expression!r} and {other.expression!r}"
            )
    return first


class UnitAwareModel(BaseModel):
    """Base model that converts unit-annotated keys before field validation.

    Supply the selected system through Pydantic's validation context, for
    example ``Model.model_validate(data, context={"unit_system": au})``.
    Annotated values are converted into that system before field types,
    constraints, and application validators run. The validator normalizes
    only this model's mapping level; nested models receive their own input and
    reuse Pydantic's validation context.

    ``UnitDimension`` metadata is translated into ``reference_units`` checks,
    so the annotation's unit expression selects dimensions but never an output
    scale. Canonical input without unit suffixes can be validated without a
    context.
    """

    # Resolved once per model class, so conflicting declarations fail when the
    # class is defined rather than on whichever input first carries a suffix.
    _unit_reference_units: ClassVar[dict[str, str]] = {}

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        reference_units: dict[str, str] = {}
        for name, model_field in cls.model_fields.items():
            dimension = _field_dimension(name, model_field)
            if dimension is not None:
                reference_units[f"/{_pointer_token(name)}"] = dimension.expression
        cls._unit_reference_units = reference_units

    @model_validator(mode="before")
    @classmethod
    def _normalize_unit_input(cls, data: Any, info: ValidationInfo) -> Any:
        if not isinstance(data, Mapping):
            return data

        # The shared mapping boundary owns suffix parsing, collision handling,
        # dimensional checks, and conversion. Do not recurse: nested models
        # need to process their own mapping level.
        raw_mapping = cast(Mapping[Any, Any], data)
        contains_annotation = any(
            isinstance(key, str) and ("[" in key or "]" in key) for key in raw_mapping
        )
        if not contains_annotation:
            return raw_mapping

        context: Any = info.context
        system: Any = None
        if isinstance(context, UnitSystem):
            system = context
        elif isinstance(context, Mapping):
            context_mapping = cast(Mapping[str, Any], context)
            system = context_mapping.get("unit_system")
        if not isinstance(system, UnitSystem):
            raise ValueError(
                "unit-annotated input requires a UnitSystem in validation context "
                "(context={'unit_system': system})"
            )

        try:
            references = parse_reference_units(cls._unit_reference_units)
            return normalize_mapping_level(
                raw_mapping,
                system,
                "$",
                recurse=False,
                references=references,
            )
        except (MappingConversionError, ReferenceUnitsError) as exc:
            # A ValueError from a before validator is represented as a normal
            # Pydantic ValidationError at the model's input location.
            raise ValueError(str(exc)) from exc


__all__ = ["UnitAwareModel"]
