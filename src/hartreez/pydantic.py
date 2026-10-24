"""Pydantic v2 integration for unit-annotated model input.

Import this module explicitly to use :class:`UnitAwareModel`; the core
``hartreez`` namespace remains independent of Pydantic.
"""

from __future__ import annotations

import types
import typing
from collections.abc import Iterator, Mapping, Sequence
from typing import Annotated, Any, Union, cast, get_args, get_origin

import typing_extensions
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


# ``type Alias = ...`` creates typing.TypeAliasType on Python 3.12+, which
# typing_extensions (a Pydantic dependency) may or may not re-export.
_TYPE_ALIAS_TYPES: tuple[type, ...] = (
    typing_extensions.TypeAliasType,
    getattr(typing, "TypeAliasType", typing_extensions.TypeAliasType),
)


def _declared_dimensions(annotation: Any) -> Iterator[UnitDimension]:
    if isinstance(annotation, _TYPE_ALIAS_TYPES):
        yield from _declared_dimensions(annotation.__value__)
        return
    origin = get_origin(annotation)
    if origin is Annotated:
        inner, *metadata = get_args(annotation)
        yield from (m for m in metadata if isinstance(m, UnitDimension))
        yield from _declared_dimensions(inner)
    elif origin in _UNION_ORIGINS or origin in _SEQUENCE_ORIGINS:
        for argument in get_args(annotation):
            yield from _declared_dimensions(argument)


def _nested_plain_models(annotation: Any) -> Iterator[type[BaseModel]]:
    if isinstance(annotation, _TYPE_ALIAS_TYPES):
        yield from _nested_plain_models(annotation.__value__)
        return
    # Check generic aliases first: on Python 3.10, list[int] passes isinstance(type).
    if get_origin(annotation) is not None:
        for argument in get_args(annotation):
            yield from _nested_plain_models(argument)
    elif (
        isinstance(annotation, type)
        and issubclass(annotation, BaseModel)
        and not issubclass(annotation, UnitAwareModel)
    ):
        yield annotation


def _describe(dimension: UnitDimension | None) -> str:
    return "none" if dimension is None else repr(dimension.expression)


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
        # Register only the input keys Pydantic accepts, so a suffixed key it
        # would ignore is rejected instead of converted and then dropped.
        config = cls.model_config
        by_alias = config.get("validate_by_alias", True)
        by_name = config.get("validate_by_name", False) or config.get(
            "populate_by_name", False
        )
        registered: dict[str, tuple[str, UnitDimension | None]] = {}
        for name, model_field in cls.model_fields.items():
            # Only UnitAwareModel converts its own level; a plain model would
            # ignore its suffixed keys and silently keep its defaults.
            for nested in _nested_plain_models(model_field.annotation):
                if any(
                    _field_dimension(nested_name, nested_field) is not None
                    for nested_name, nested_field in nested.model_fields.items()
                ):
                    raise TypeError(
                        f"field {name!r} nests {nested.__name__}, which declares "
                        f"UnitDimension fields but is not a UnitAwareModel; derive "
                        f"{nested.__name__} from UnitAwareModel"
                    )
            # Fields without a dimension are registered too, so a key they share
            # with a dimensioned field cannot hand them a converted value.
            dimension = _field_dimension(name, model_field)
            # Input arrives under the validation alias, which Pydantic also
            # fills from ``alias`` and ``alias_generator``. AliasPath and
            # AliasChoices are not supported.
            alias = model_field.validation_alias
            keys: list[str] = []
            if alias is None or by_name:
                keys.append(name)
            if isinstance(alias, str) and by_alias:
                keys.append(alias)
            for key in keys:
                pointer = f"/{_pointer_token(key)}"
                previous_name, previous = registered.setdefault(pointer, (name, dimension))
                # Pydantic fills every field that shares an input key, so one
                # conversion factor must suit all of them.
                previous_dimensions = None if previous is None else previous.dimensions
                dimensions = None if dimension is None else dimension.dimensions
                if previous_dimensions != dimensions:
                    raise TypeError(
                        f"input key {key!r} is shared by fields {previous_name!r} and "
                        f"{name!r} with conflicting UnitDimension metadata: "
                        f"{_describe(previous)} and {_describe(dimension)}"
                    )
        return {
            pointer: dimension.expression
            for pointer, (_, dimension) in registered.items()
            if dimension is not None
        }

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
