"""Small, dependency-free SI unit parsing and conversion for numeric code.

Importing :mod:`hartreez` only loads its small standard-library registry. No
optional numerical or validation packages are imported.
"""

from __future__ import annotations

from typing import Any

from hartreez.constants import constant_names
from hartreez.dimensions import Dimensions
from hartreez.errors import (
    IncompatibleUnitsError,
    MappingConversionError,
    ReferenceUnitsError,
    UnitError,
    UnitSyntaxError,
    UnitSystemError,
    UnknownUnitError,
)
from hartreez.mapping import convert_mapping
from hartreez.parser import parse_unit
from hartreez.systems import UnitSystem, au
from hartreez.units import Unit, canonical_unit_names, unit_aliases, unit_names


def convert(value: Any, from_unit: str | Unit, to_unit: str | Unit) -> Any:
    """Convert ``value`` between compatible units by multiplying by a float.

    Values are not wrapped or coerced. The caller's value must support
    multiplication by a Python ``float``.
    """

    source = parse_unit(from_unit) if isinstance(from_unit, str) else from_unit
    destination = parse_unit(to_unit) if isinstance(to_unit, str) else to_unit
    if type(source) is not Unit:
        raise TypeError("from_unit must be a unit expression or Unit")
    if type(destination) is not Unit:
        raise TypeError("to_unit must be a unit expression or Unit")
    if source.dimensions != destination.dimensions:
        raise IncompatibleUnitsError(
            f"cannot convert from dimensions {source.dimensions} to {destination.dimensions}"
        )
    return value * (source.scale / destination.scale)


def main() -> int:
    """Run the standard-library command line interface."""

    from hartreez.cli import main as cli_main

    return cli_main()


__all__ = [
    "Dimensions",
    "IncompatibleUnitsError",
    "MappingConversionError",
    "ReferenceUnitsError",
    "Unit",
    "UnitError",
    "UnitSyntaxError",
    "UnitSystem",
    "UnitSystemError",
    "UnknownUnitError",
    "au",
    "canonical_unit_names",
    "constant_names",
    "convert",
    "convert_mapping",
    "main",
    "parse_unit",
    "unit_aliases",
    "unit_names",
]
