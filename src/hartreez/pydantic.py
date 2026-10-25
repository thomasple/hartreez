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
_SEQUENCE_ORIGINS: tuple[Any, ...] = (list, set, frozenset, Sequence)
_NONE_TYPE = type(None)


# ``type Alias = ...`` creates typing.TypeAliasType on Python 3.12+, which
# typing_extensions (a Pydantic dependency) may or may not re-export.
_TYPE_ALIAS_TYPES: tuple[type, ...] = (
    typing_extensions.TypeAliasType,
    getattr(typing, "TypeAliasType", typing_extensions.TypeAliasType),
)


def _is_type_alias(annotation: Any) -> bool:
    # On Python 3.10, a generic alias such as ``Vector[Length]`` also passes the
    # isinstance check, but its __value__ still holds the unsubstituted
    # parameters, so only an alias without arguments is expanded.
    return isinstance(annotation, _TYPE_ALIAS_TYPES) and get_origin(annotation) is None


def _alias_arguments(annotation: Any) -> tuple[Any, ...]:
    # The arguments of an annotation, plus the body of a generic type alias.
    arguments = get_args(annotation)
    origin = get_origin(annotation)
    if isinstance(origin, _TYPE_ALIAS_TYPES):
        arguments += (origin.__value__,)
    return arguments


def _contains_unit_dimension(annotation: Any) -> bool:
    """Whether ``annotation`` contains UnitDimension metadata anywhere.

    This is only a guard for unsupported annotation shapes: supported shapes are
    interpreted by ``_project_dimension`` below. It deliberately understands no
    semantics beyond finding the marker.
    """

    if _is_type_alias(annotation):
        return _contains_unit_dimension(annotation.__value__)
    origin = get_origin(annotation)
    if origin is Annotated:
        inner, *metadata = get_args(annotation)
        return any(isinstance(item, UnitDimension) for item in metadata) or (
            _contains_unit_dimension(inner)
        )
    return any(_contains_unit_dimension(argument) for argument in _alias_arguments(annotation))


def _merge_dimensions(
    name: str, dimensions: list[UnitDimension]
) -> UnitDimension | None:
    if not dimensions:
        return None
    first = dimensions[0]
    for other in dimensions[1:]:
        if other.dimensions != first.dimensions:
            raise TypeError(
                f"field {name!r} declares conflicting UnitDimension metadata: "
                f"{first.expression!r} and {other.expression!r}"
            )
    return first


def _project_dimension(name: str, annotation: Any) -> UnitDimension | None:
    """Project supported dimensional annotations onto their containing field.

    A unit suffix belongs to the field's input key, so a nested UnitDimension is
    useful only when the whole field has one unambiguous dimension. Supported
    structure is intentionally small: Annotated, type aliases, Optional[T],
    homogeneous list/set/frozenset/Sequence containers, and homogeneous tuples.
    Any other shape containing UnitDimension metadata is rejected loudly.
    """

    if _is_type_alias(annotation):
        return _project_dimension(name, annotation.__value__)

    origin = get_origin(annotation)
    if origin is Annotated:
        inner, *metadata = get_args(annotation)
        dimensions = [item for item in metadata if isinstance(item, UnitDimension)]
        inner_dimension = _project_dimension(name, inner)
        if inner_dimension is not None:
            dimensions.append(inner_dimension)
        return _merge_dimensions(name, dimensions)

    if origin in _UNION_ORIGINS:
        arguments = get_args(annotation)
        non_none = [argument for argument in arguments if argument is not _NONE_TYPE]
        if len(non_none) == 1 and len(non_none) != len(arguments):
            return _project_dimension(name, non_none[0])
        if _contains_unit_dimension(annotation):
            raise TypeError(
                f"field {name!r} contains UnitDimension inside an unsupported union; "
                "only Optional[T] is supported"
            )
        return None

    if origin in _SEQUENCE_ORIGINS:
        arguments = get_args(annotation)
        if not arguments:
            return None
        return _project_dimension(name, arguments[0])

    if origin is tuple:
        arguments = get_args(annotation)
        if not arguments:
            return None
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return _project_dimension(name, arguments[0])
        projected = [_project_dimension(name, argument) for argument in arguments]
        present = [dimension for dimension in projected if dimension is not None]
        if not present:
            return None
        if len(present) != len(projected):
            raise TypeError(
                f"field {name!r} mixes dimensioned and undimensioned tuple elements"
            )
        return _merge_dimensions(name, present)

    if _contains_unit_dimension(annotation):
        raise TypeError(
            f"field {name!r} contains UnitDimension inside unsupported annotation "
            f"{annotation!r}"
        )
    return None


def _nested_plain_models(annotation: Any) -> Iterator[type[BaseModel]]:
    if _is_type_alias(annotation):
        yield from _nested_plain_models(annotation.__value__)
        return
    # Check generic aliases first: on Python 3.10, list[int] passes isinstance(type).
    if get_origin(annotation) is not None:
        for argument in _alias_arguments(annotation):
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
    # Pydantic lifts top-level Annotated metadata into FieldInfo.metadata. Nested
    # aliases and containers remain in model_field.annotation and are projected
    # onto the containing field by the deliberately small grammar above.
    dimensions = [
        item for item in model_field.metadata if isinstance(item, UnitDimension)
    ]
    projected = _project_dimension(name, model_field.annotation)
    if projected is not None:
        dimensions.append(projected)
    return _merge_dimensions(name, dimensions)


class UnitAwareModel(BaseModel):
    """Base model that converts unit-annotated keys before field validation.

    Supply the selected system through Pydantic's validation context, for
    example ``Model.model_validate(data, context={"unit_system": au})``.
    Annotated values are converted into that system before field types,
    constraints, and application validators run. The validator normalizes
    only this model's mapping level; nested models receive their own input and
    reuse Pydantic's validation context.

    ``UnitDimension`` metadata is projected onto its containing field, then
    translated into ``reference_units`` checks. The annotation's unit expression
    selects dimensions but never an output scale. A suffixed key must name a
    field with one unambiguous projected dimension, so no value is converted
    without a dimensional check. Canonical input without unit suffixes can be
    validated without a context.
    """

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        # Resolve once here so conflicting or unsupported declarations fail when
        # the class is defined rather than on whichever input first has a suffix.
        # Resolve again during validation because model_rebuild() can complete
        # forward-referenced annotations later.
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
