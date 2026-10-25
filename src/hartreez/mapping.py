"""Normalization of unit-annotated Python mappings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast, overload

from hartreez.errors import MappingConversionError, ReferenceUnitsError, UnitError
from hartreez.parser import parse_unit
from hartreez.systems import UnitSystem
from hartreez.units import Unit


_Reference = tuple[str, Unit]


def _pointer_child(path: str | None, component: str | int) -> str | None:
    """Extend a reference path, or keep it unavailable below a non-string key."""

    if path is None:
        return None
    escaped = str(component).replace("~", "~0").replace("/", "~1")
    return f"{path}/{escaped}"


def parse_reference_units(reference_units: Mapping[str, str] | None) -> dict[str, _Reference]:
    """Parse and validate the complete schema before inspecting input values."""

    raw: Any = reference_units
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ReferenceUnitsError("reference_units must be a mapping of JSON Pointer paths to unit expressions")

    parsed: dict[str, _Reference] = {}
    for raw_path, expression in cast(Mapping[Any, Any], raw).items():
        if not isinstance(raw_path, str):
            raise ReferenceUnitsError(
                f"invalid reference path {raw_path!r}; expected a JSON Pointer path string"
            )
        if raw_path == "":
            # Reference paths address fields, not the document root. Use the
            # empty string as the slashless spelling of the empty root key.
            path = "/"
        elif raw_path.startswith("/"):
            path = raw_path
        else:
            # JSON Pointer paths may omit their leading slash. The remaining
            # tokens and their ~0/~1 escapes keep the same meaning.
            path = f"/{raw_path}"
        # JSON Pointer permits only these two escapes. Reject malformed paths
        # early so a typo cannot silently leave a field without validation.
        index = 0
        while index < len(path):
            if path[index] == "~":
                if path[index : index + 2] not in ("~0", "~1"):
                    raise ReferenceUnitsError(f"invalid JSON Pointer escape in reference path {path!r}")
                index += 2
            else:
                index += 1
        if not isinstance(expression, str):
            raise ReferenceUnitsError(
                f"invalid reference definition at {raw_path}: expected a unit expression string, "
                f"got {type(expression).__name__}"
            )
        try:
            unit = parse_unit(expression)
        except (UnitError, TypeError) as exc:
            raise ReferenceUnitsError(
                f"invalid reference unit expression {expression!r} at {raw_path}: {exc}"
            ) from exc
        if path in parsed:
            raise ReferenceUnitsError(f"duplicate reference path {path!r}")
        parsed[path] = (expression, unit)
    return parsed


def _path_child(path: str, key: object) -> str:
    """Return a readable normalized path for a mapping key."""

    if isinstance(key, str) and key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{key!r}]"


def _annotation(key: str, path: str) -> tuple[str, str] | None:
    """Split one strict terminal ``[unit expression]`` annotation."""

    has_bracket = "[" in key or "]" in key
    if not has_bracket:
        return None
    if not key.endswith("]"):
        raise MappingConversionError(
            f"malformed unit annotation in key {key!r} at {path}; "
            "expected a terminal '[unit expression]' suffix"
        )
    open_bracket = key.rfind("[")
    if open_bracket < 0 or "[" in key[:open_bracket] or "]" in key[:open_bracket]:
        raise MappingConversionError(
            f"malformed unit annotation in key {key!r} at {path}; "
            "expected a key followed by one terminal '[unit expression]' suffix"
        )
    expression = key[open_bracket + 1 : -1]
    if not expression.strip():
        raise MappingConversionError(
            f"malformed unit annotation in key {key!r} at {path}; unit expression is empty"
        )
    canonical = key[:open_bracket]
    if canonical != canonical.strip():
        raise MappingConversionError(
            f"malformed unit annotation in key {key!r} at {path}; "
            "the canonical key cannot have surrounding whitespace"
        )
    return canonical, expression


def _scale_value(value: Any, factor: float, path: str, original_key: str) -> Any:
    """Scale supported annotated values, retaining list and tuple structure."""

    if isinstance(value, Mapping):
        raise MappingConversionError(
            f"annotated key {original_key!r} at {path} has a mapping value; "
            "unit annotations cannot be applied to mappings"
        )
    if isinstance(value, list):
        items = cast(list[Any], value)
        return [
            _scale_value(item, factor, f"{path}[{index}]", original_key)
            for index, item in enumerate(items)
        ]
    if isinstance(value, tuple):
        items = cast(tuple[Any, ...], value)
        return tuple(
            _scale_value(item, factor, f"{path}[{index}]", original_key)
            for index, item in enumerate(items)
        )
    if isinstance(value, bool) or value is None or isinstance(value, str):
        raise MappingConversionError(
            f"cannot scale value at {path} from annotated key {original_key!r}: "
            f"unsupported value of type {type(value).__name__}"
        )
    try:
        return value * factor
    except Exception as exc:
        raise MappingConversionError(
            f"cannot scale value at {path} from annotated key {original_key!r}: "
            f"unsupported value of type {type(value).__name__} "
            "(it must support multiplication by a Python float)"
        ) from exc


def normalize_mapping_level(
    mapping: Mapping[Any, Any],
    system: UnitSystem,
    path: str,
    *,
    references: Mapping[str, _Reference] | None = None,
    pointer: str | None = "",
) -> dict[Any, Any]:
    """Normalize one mapping level and recurse into unannotated values."""

    entries: list[tuple[Any, Any, str, str | None, str | None]] = []
    seen: dict[Any, Any] = {}
    for key, value in mapping.items():
        child_path = _path_child(path, key)
        if isinstance(key, str):
            annotation = _annotation(key, child_path)
        else:
            annotation = None
        canonical, expression = annotation if annotation is not None else (key, None)
        if canonical in seen:
            first = seen[canonical]
            raise MappingConversionError(
                f"normalized key collision at {_path_child(path, canonical)}: "
                f"keys {first!r} and {key!r} both normalize to {canonical!r}"
            )
        seen[canonical] = key
        entries.append((canonical, value, child_path, expression, key if expression is not None else None))

    normalized: dict[Any, Any] = {}
    for canonical, value, original_path, expression, original_key in entries:
        target_path = _path_child(path, canonical)
        # JSON Pointer has no type information for object keys. Support string
        # mapping keys and array indices; do not alias e.g. key 1 with key "1".
        target_pointer = _pointer_child(pointer, canonical) if isinstance(canonical, str) else None
        if expression is not None:
            if original_key is None:
                raise RuntimeError("annotated mapping entry has no original key")
            try:
                unit = parse_unit(expression)
            except UnitError as exc:
                raise MappingConversionError(
                    f"invalid unit expression {expression!r} in key {original_key!r} "
                    f"at {original_path}: {exc}"
                ) from exc
            reference = (
                references.get(target_pointer)
                if references is not None and target_pointer is not None
                else None
            )
            if reference is not None:
                reference_expression, reference_unit = reference
                if unit.dimensions != reference_unit.dimensions:
                    raise MappingConversionError(
                        f"dimensional mismatch at normalized path {target_pointer}: supplied unit "
                        f"{expression!r} has dimensions {unit.dimensions}, but reference unit "
                        f"{reference_expression!r} has dimensions {reference_unit.dimensions}"
                    )
            try:
                factor = system.factor_from(unit)
            except Exception as exc:
                raise MappingConversionError(
                    f"could not convert key {original_key!r} with unit expression "
                    f"{expression!r} at {original_path}: {exc}"
                ) from exc
            normalized[canonical] = _scale_value(value, factor, target_path, original_key)
        else:
            normalized[canonical] = _normalize_unannotated(
                value, system, target_path, references, target_pointer
            )
    return normalized


def _normalize_unannotated(
    value: Any,
    system: UnitSystem,
    path: str,
    references: Mapping[str, _Reference] | None,
    pointer: str | None,
) -> Any:
    if isinstance(value, Mapping):
        return normalize_mapping_level(
            cast(Mapping[Any, Any], value), system, path, references=references, pointer=pointer
        )
    if isinstance(value, list):
        items = cast(list[Any], value)
        return [
            _normalize_unannotated(item, system, f"{path}[{index}]", references, _pointer_child(pointer, index))
            for index, item in enumerate(items)
        ]
    if isinstance(value, tuple):
        items = cast(tuple[Any, ...], value)
        return tuple(
            _normalize_unannotated(item, system, f"{path}[{index}]", references, _pointer_child(pointer, index))
            for index, item in enumerate(items)
        )
    return value


@overload
def convert_mapping(
    mapping: Mapping[Any, Any], system: UnitSystem, *, reference_units: Mapping[str, str] | None = None
) -> dict[Any, Any]: ...


@overload
def convert_mapping(mapping: Any, system: Any, *, reference_units: Any = None) -> dict[Any, Any]: ...


def convert_mapping(mapping: Any, system: Any, *, reference_units: Any = None) -> dict[Any, Any]:
    """Return a new mapping with annotated values converted into ``system``.

    String keys ending in ``[unit expression]`` are converted and returned
    without that suffix. Other keys and values are treated as already in the
    supplied system. Nested mappings are normalized recursively. The input is
    never mutated and no serialization framework is required.

    Annotated lists and tuples are scaled recursively. Other scalar or
    array-like values are passed through ``value * float_factor`` so numerical
    frameworks can participate without becoming dependencies of this package.

    ``reference_units`` optionally maps JSON Pointer paths to expected
    dimensions; the leading slash may be omitted. For example,
    ``{"dt": "fs", "simulation/dt": "fs"}``. Pointers
    address string mapping keys and zero-based list/tuple indices. References
    below any non-string mapping key are not addressable and are ignored there,
    since JSON Pointer cannot distinguish that key from a string key with the
    same spelling. References are validation-only; annotated values are still
    converted into ``system``.
    """

    if not isinstance(mapping, Mapping):
        raise TypeError("mapping must implement collections.abc.Mapping")
    if not isinstance(system, UnitSystem):
        raise TypeError("system must be a UnitSystem")
    references = parse_reference_units(reference_units)
    return normalize_mapping_level(
        cast(Mapping[Any, Any], mapping), system, "$", references=references
    )
