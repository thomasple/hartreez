"""Pydantic v2 integration for unit-annotated model input.

Import this module explicitly to use :class:`UnitAwareModel`; the core
``hartreez`` namespace remains independent of Pydantic.
"""

from __future__ import annotations

import types
import typing
from collections import Counter, deque
from contextvars import ContextVar
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import TYPE_CHECKING, Annotated, Any, Union, cast, get_args, get_origin

import typing_extensions
from pydantic import AliasChoices, AliasPath, BaseModel, ValidationInfo, model_validator
from pydantic.fields import FieldInfo

from hartreez.errors import MappingConversionError, ReferenceUnitsError
from hartreez.mapping import normalize_mapping_level, parse_reference_units
from hartreez.metadata import SCHEMA_METADATA_KEY, UnitDimension
from hartreez.systems import UnitSystem

if TYPE_CHECKING:
    # Exported by Pydantic 2.10 and later; only needed for type checking.
    from pydantic import ModelWrapValidatorHandler


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


_UNION_ORIGINS: tuple[Any, ...] = (Union, types.UnionType)
_SEQUENCE_ORIGINS: tuple[Any, ...] = (list, set, frozenset, Sequence)
_NONE_TYPE = type(None)
# Runtime containers searched for suffixed keys left after validation.
_SCANNED_CONTAINERS: tuple[type, ...] = (list, tuple, set, frozenset, deque)


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


def _type_arguments(annotation: Any) -> tuple[Any, ...]:
    """Return an annotation's arguments, plus the body of a generic type alias."""

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
    return any(_contains_unit_dimension(argument) for argument in _type_arguments(annotation))


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


def _has_unit_suffix(key: Any) -> bool:
    """Whether ``key`` is a string the mapping layer would read as annotated."""

    # Like the mapping layer, treat any bracket as an attempted annotation so a
    # malformed suffix is reported rather than kept as part of the key.
    return isinstance(key, str) and ("[" in key or "]" in key)


def _model_items(model: BaseModel) -> dict[str, Any]:
    """Return a model's field values together with its extra values."""

    return {**model.__dict__, **(model.__pydantic_extra__ or {})}


def _child_path(path: str, key: Any) -> str:
    if isinstance(key, str) and key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{key!r}]"


@dataclass
class _PendingMapping:
    """A raw input mapping holding a suffixed key, which models must convert."""

    mapping: Mapping[Any, Any]
    # Path of its first occurrence in the input, and its first suffixed key.
    path: str
    key: str
    # How often the mapping object appears in the input. Each occurrence
    # needs its own UnitAwareModel in the result.
    occurrences: int = 0
    # Occurrences not yet matched to a nested model, by identity or as a copy.
    unmatched: int = 0


# Shared empty result of _SuffixLedger._record; never updated.
_NO_COUNTS: Counter[int] = Counter()

# Suffixed key set -> signature of their values -> pending mappings by id.
_Buckets = dict[frozenset[str], dict[Hashable, dict[int, _PendingMapping]]]


@dataclass
class _SuffixLedger:
    """Suffixed raw mappings below one root UnitAwareModel, and who converted them.

    Pydantic hands every nested model the raw mapping object from the input,
    so object identity links a raw mapping to the model built from it. A
    validator may hand on a copy instead, possibly with keys added or
    removed, which is matched by its unchanged suffixed entries.
    """

    # Pending mappings by id.
    pending: dict[int, _PendingMapping] = field(default_factory=dict[int, _PendingMapping])
    # Pending mappings with unmatched occurrences, for matching copies.
    buckets: _Buckets = field(default_factory=_Buckets)
    # id of each copy -> id of the pending mapping it was matched to, if any.
    copies: dict[int, int | None] = field(default_factory=dict[int, int | None])
    # id of each nested UnitAwareModel instance -> id of the pending mapping
    # it was built from.
    sources: dict[int, int] = field(default_factory=dict[int, int])
    # Keeps recorded objects alive so their ids stay unique until the check.
    keep: list[Any] = field(default_factory=list[Any])

    def record(self, data: Mapping[Any, Any]) -> None:
        """Record every suffixed mapping below the root input ``data``."""

        counts = Counter[int]()
        # The root's own keys are converted or rejected by its before hook.
        ancestors = {id(data)}
        memo: dict[int, Counter[int]] = {}
        for key, value in data.items():
            counts.update(self._record(value, _child_path("$", key), ancestors, memo))
        for mapping_id, occurrences in counts.items():
            entry = self.pending[mapping_id]
            entry.occurrences = entry.unmatched = occurrences
            self._bucket(entry.mapping, _suffixed_keys(entry.mapping))[mapping_id] = entry

    def _record(
        self, value: Any, path: str, ancestors: set[int], memo: dict[int, Counter[int]]
    ) -> Counter[int]:
        # Returns how often each pending mapping occurs below ``value``. A
        # value shared by several parents is walked once and counted at each
        # of them. Paths and counters are only built where needed, since most
        # containers in numeric input hold only numbers.
        value_id = id(value)
        if value_id in memo:
            return memo[value_id]
        if value_id in ancestors:
            return _NO_COUNTS
        counts: Counter[int] | None = None
        if isinstance(value, Mapping):
            mapping = cast(Mapping[Any, Any], value)
            suffixed = next((key for key in mapping if _has_unit_suffix(key)), None)
            if suffixed is not None:
                self.pending[value_id] = _PendingMapping(mapping, path, suffixed)
                self.keep.append(mapping)
                counts = Counter({value_id: 1})
            children = cast(Iterable[tuple[Any, Any]], mapping.items())
            keyed = True
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            # Any sequence, including a deque, can hold nested mappings.
            # Other iterables are not iterated, since that could consume them.
            children = enumerate(cast(Sequence[Any], value))
            keyed = False
        else:
            return _NO_COUNTS
        ancestors.add(value_id)
        for key, child in children:
            if isinstance(child, (float, int, str)):
                continue
            child_path = _child_path(path, key) if keyed else f"{path}[{key}]"
            found = self._record(child, child_path, ancestors, memo)
            if not found:
                continue
            if counts is None:
                counts = Counter(found)
            else:
                counts.update(found)
        ancestors.discard(value_id)
        result = _NO_COUNTS if counts is None else counts
        memo[value_id] = result
        return result

    def source_of(self, data: Any) -> int | None:
        """Return the id of the pending mapping ``data`` is, or is a copy of."""

        data_id = id(data)
        if data_id in self.pending:
            self._claim(self.pending[data_id], data_id)
            return data_id
        if data_id not in self.copies:
            self.copies[data_id] = self._match_copy(data)
            self.keep.append(data)
        return self.copies[data_id]

    def unconverted(self, result: Any) -> tuple[_PendingMapping, int] | None:
        """Return a pending mapping with fewer models in ``result`` than occurrences.

        The count returned is how many models in the result were built from it.
        """

        converted = Counter[int]()
        _count_sources(result, self, converted, set())
        for mapping_id, entry in self.pending.items():
            if converted[mapping_id] < entry.occurrences:
                return entry, converted[mapping_id]
        return None

    def _claim(self, entry: _PendingMapping, mapping_id: int) -> None:
        # Once every occurrence is matched, no copy can claim the mapping.
        entry.unmatched -= 1
        if entry.unmatched <= 0:
            self._bucket(entry.mapping, _suffixed_keys(entry.mapping)).pop(mapping_id, None)

    def _bucket(self, mapping: Mapping[Any, Any], keys: frozenset[str]) -> dict[int, _PendingMapping]:
        # Unhashable values share one bucket, searched by comparison.
        return self.buckets.setdefault(keys, {}).setdefault(_signature(mapping, keys), {})

    def _match_copy(self, data: Any) -> int | None:
        # A copy holds every suffixed entry of the pending mapping it came
        # from, unchanged; other keys may have been added or removed. Each
        # occurrence of a pending mapping is matched by at most one copy,
        # preferring the mapping with the most suffixed entries.
        if not isinstance(data, Mapping):
            return None
        mapping = cast(Mapping[Any, Any], data)
        keys = _suffixed_keys(mapping)
        if not keys:
            return None
        groups = [group for group in self.buckets if group <= keys]
        for group in sorted(groups, key=len, reverse=True):
            for candidate_id, entry in list(self._bucket(mapping, group).items()):
                if all(_same_value(entry.mapping[key], mapping[key]) for key in group):
                    self._claim(entry, candidate_id)
                    return candidate_id
        return None


def _suffixed_keys(mapping: Mapping[Any, Any]) -> frozenset[str]:
    return frozenset(key for key in mapping if _has_unit_suffix(key))


def _frozen(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in cast(Iterable[Any], value))
    return value


def _signature(mapping: Mapping[Any, Any], keys: frozenset[str]) -> Hashable:
    """Return a hashable summary of the values under ``keys``, or None."""

    signature = tuple(_frozen(mapping[key]) for key in sorted(keys))
    try:
        hash(signature)
    except TypeError:
        return None
    return signature


def _same_value(left: Any, right: Any) -> bool:
    """Whether a suffixed value is unchanged, including its type."""

    if left is right:
        return True
    # True == 1.0 in Python, but a value of another type has been changed.
    if type(left) is not type(right):
        return False
    if isinstance(left, (list, tuple)):
        items = cast(Sequence[Any], left)
        others = cast(Sequence[Any], right)
        return len(items) == len(others) and all(map(_same_value, items, others))
    try:
        return bool(left == right)
    except Exception:
        # e.g. NumPy arrays, whose comparison has no single truth value.
        return False


_LEDGER: ContextVar[_SuffixLedger | None] = ContextVar("hartreez_suffix_ledger", default=None)


def _count_sources(value: Any, ledger: _SuffixLedger, converted: Counter[int], seen: set[int]) -> None:
    """Count the pending mappings the UnitAwareModels in a validated value came from."""

    if id(value) in seen:
        return
    seen.add(id(value))
    children: Iterable[Any]
    if isinstance(value, BaseModel):
        if isinstance(value, UnitAwareModel) and id(value) in ledger.sources:
            converted[ledger.sources[id(value)]] += 1
        children = _model_items(value).values()
    elif is_dataclass(value) and not isinstance(value, type):
        children = (getattr(value, item.name) for item in fields(value))
    elif isinstance(value, Mapping):
        children = cast(Mapping[Any, Any], value).values()
    elif isinstance(value, _SCANNED_CONTAINERS):
        children = cast(Iterable[Any], value)
    else:
        return
    for child in children:
        if not isinstance(child, (float, int, str)):
            _count_sources(child, ledger, converted, seen)


# Core schema nodes that validate a class from a mapping, keeping only the
# keys that class declares.
_SCHEMA_CLASS_NODES = ("model", "dataclass", "typed-dict")


def _schema_children(node: Mapping[str, Any]) -> Iterator[tuple[str | None, Any]]:
    """Yield the child schemas of a core schema node, with field names."""

    node_type = node.get("type")
    for key, child in node.items():
        if key == "fields" and node_type in ("model-fields", "typed-dict"):
            for name, field_schema in cast(Mapping[str, Any], child).items():
                yield name, field_schema
        elif key == "fields" and node_type == "dataclass-args":
            for field_schema in cast(list[Mapping[str, Any]], child):
                yield field_schema["name"], field_schema
        else:
            yield None, child


def _plain_class_with_dimension(
    schema: Any, owner: type | None, path: list[str]
) -> tuple[type, list[str]] | None:
    """Find a UnitDimension mark below a plain model, dataclass or TypedDict.

    Those classes drop suffixed keys as unknown keys, so a dimension declared
    inside one would silently keep its default. Pydantic has already resolved
    aliases, generics and forward references in the compiled schema.
    """

    if isinstance(schema, list):
        for item in cast(list[Any], schema):
            found = _plain_class_with_dimension(item, owner, path)
            if found is not None:
                return found
        return None
    if not isinstance(schema, Mapping):
        return None
    node = cast(Mapping[str, Any], schema)
    # A mark on a class node annotates the field holding it, so check it with
    # the enclosing owner before the node's own class takes over.
    metadata = node.get("metadata")
    if owner is not None and isinstance(metadata, Mapping) and SCHEMA_METADATA_KEY in metadata:
        return owner, path
    schema_class = node.get("cls")
    if node.get("type") in _SCHEMA_CLASS_NODES and isinstance(schema_class, type):
        # A UnitAwareModel converts the suffixes of its own fields.
        owner = None if issubclass(schema_class, UnitAwareModel) else cast(type[Any], schema_class)
    for name, child in _schema_children(node):
        found = _plain_class_with_dimension(child, owner, path if name is None else [*path, name])
        if found is not None:
            return found
    return None


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
        # Fail when the class is defined rather than on whichever input first
        # has a suffix. A model with unresolved forward references has no
        # compiled schema yet; its suffixed input is still checked when it
        # validates.
        if cls.__pydantic_complete__:
            found = _plain_class_with_dimension(cls.__pydantic_core_schema__, None, [])
            if found is not None:
                nested, path = found
                raise TypeError(
                    f"field {'.'.join(path)!r} nests {nested.__name__}, which declares "
                    "UnitDimension fields but is not a UnitAwareModel; make "
                    f"{nested.__name__} a UnitAwareModel"
                )
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
            # Fields without a dimension are registered too, so a key they share
            # with a dimensioned field cannot hand them a converted value.
            dimension = _field_dimension(name, model_field)
            # Input arrives under the validation alias, which Pydantic also
            # fills from ``alias`` and ``alias_generator``.
            alias = model_field.validation_alias
            choices: list[str | AliasPath] = []
            if isinstance(alias, AliasChoices):
                choices = alias.choices
            elif alias is not None:
                choices = [alias]
            if dimension is not None and any(isinstance(c, AliasPath) for c in choices):
                raise TypeError(
                    f"field {name!r} declares UnitDimension metadata and an AliasPath "
                    "validation alias; unit suffixes are only read at the model's own "
                    "mapping level, so use a string alias"
                )
            keys: list[str] = []
            if alias is None or by_name:
                keys.append(name)
            if by_alias:
                for choice in choices:
                    # An AliasPath reads below this level, but its first key is
                    # an input key here and can be shared like any other.
                    first = choice if isinstance(choice, str) else choice.path[0]
                    if isinstance(first, str):
                        keys.append(first)
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
        if not any(_has_unit_suffix(key) for key in raw_mapping):
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

    @model_validator(mode="wrap")
    @classmethod
    def _require_every_suffix_converted(
        cls, data: Any, handler: ModelWrapValidatorHandler[UnitAwareModel]
    ) -> UnitAwareModel:
        # Every suffixed key in the input must be converted by a UnitAwareModel
        # that ends up in the result. A key that a dict field kept, or that a
        # plain model, dataclass or TypedDict dropped, has no such model.
        ledger = _LEDGER.get()
        if ledger is not None:
            instance = handler(data)
            source = ledger.source_of(data)
            if source is not None:
                ledger.sources[id(instance)] = source
                ledger.keep.append(instance)
            return instance

        ledger = _SuffixLedger()
        if isinstance(data, Mapping):
            ledger.record(cast(Mapping[Any, Any], data))
        token = _LEDGER.set(ledger)
        try:
            instance = handler(data)
        finally:
            _LEDGER.reset(token)
        if not ledger.pending:
            return instance

        found = ledger.unconverted(instance)
        if found is not None:
            entry, converted = found
            message = (
                f"key {entry.key!r} at {entry.path} has a unit suffix, but no "
                "UnitAwareModel in the result converted it"
            )
            if entry.occurrences > 1:
                message = (
                    f"key {entry.key!r} at {entry.path} has a unit suffix, and its "
                    f"mapping appears {entry.occurrences} times in the input, but only "
                    f"{converted} UnitAwareModel(s) in the result converted it"
                )
            raise ValueError(message)
        return instance


__all__ = ["UnitAwareModel"]
