"""Pydantic v2 integration for unit-annotated model input.

Import this module explicitly to use :class:`UnitAwareModel`; the core
``hartreez`` namespace remains independent of Pydantic.
"""

from __future__ import annotations

import types
from collections.abc import Iterator, Mapping, Sequence
from typing import Annotated, Any, Union, cast, get_args, get_origin

from pydantic import BaseModel, ValidationInfo, model_validator
from pydantic.fields import FieldInfo

from hartreez.errors import MappingConversionError, ReferenceUnitsError
from hartreez.mapping import normalize_mapping_level, parse_reference_units
from hartreez.metadata import UnitDimension
from hartreez.systems import UnitSystem


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


_UNION_ORIGINS: tuple[Any, ...] = (Union, types.UnionType)
# One suffix scales every element of a list or tuple value, so a dimension
# declared on the element type is the dimension of the whole field.
_SEQUENCE_ORIGINS: tuple[Any, ...] = (list, tuple, set, frozenset, Sequence)


def _declared_dimensions(annotation: Any) -> Iterator[UnitDimension]:
    origin = get_origin(annotation)
    if origin is Annotated:
        inner, *metadata = get_args(annotation)
        yield from (m for m in metadata if isinstance(m, UnitDimension))
        yield from _declared_dimensions(inner)
    elif origin in _UNION_ORIGINS or origin in _SEQUENCE_ORIGINS:
        for argument in get_args(annotation):
            yield from _declared_dimensions(argument)


def _field_dimension(name: str, model_field: FieldInfo) -> UnitDimension | None:
    # Pydantic only lifts Annotated metadata from the top of an annotation, so
    # dimensions on union members such as ``Length | None`` or on element types
    # such as ``list[Length]`` stay inside ``model_field.annotation``.
    candidates = [m for m in model_field.metadata if isinstance(m, UnitDimension)]
    candidates.extend(_declared_dimensions(model_field.annotation))

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
    scale. A suffixed key must name a field that declares a ``UnitDimension``,
    so no value is converted without a dimensional check. Canonical input
    without unit suffixes can be validated without a context.
    """

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        # Resolve once here so conflicting declarations fail when the class is
        # defined rather than on whichever input first carries a suffix. The
        # validator resolves again because model_rebuild() can still complete
        # forward-referenced annotations.
        cls._unit_reference_units()

    @classmethod
    def _unit_reference_units(cls) -> dict[str, str]:
        reference_units: dict[str, str] = {}
        for name, model_field in cls.model_fields.items():
            dimension = _field_dimension(name, model_field)
            if dimension is None:
                continue
            # Input arrives under the validation alias, which Pydantic also
            # fills from ``alias`` and ``alias_generator``. AliasPath and
            # AliasChoices are not supported.
            keys = [name]
            if isinstance(model_field.validation_alias, str):
                keys.append(model_field.validation_alias)
            for key in keys:
                reference_units.setdefault(f"/{_pointer_token(key)}", dimension.expression)
        return reference_units

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
            references = parse_reference_units(cls._unit_reference_units())
            return normalize_mapping_level(
                raw_mapping,
                system,
                "$",
                recurse=False,
                references=references,
                require_references=True,
            )
        except (MappingConversionError, ReferenceUnitsError) as exc:
            # A ValueError from a before validator is represented as a normal
            # Pydantic ValidationError at the model's input location.
            raise ValueError(str(exc)) from exc


__all__ = ["UnitAwareModel"]
