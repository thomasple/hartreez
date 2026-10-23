"""Coherent, immutable working unit systems."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeVar, overload

from hartreez import constants
from hartreez.dimensions import Dimensions
from hartreez.errors import UnitSystemError
from hartreez.parser import parse_unit
from hartreez.units import CHARGE, ENERGY, LENGTH, MASS, TEMPERATURE, TIME, Unit


_V = TypeVar("_V", bound="_StoredValue")


class _StoredValue:
    """A float computed once per system and read from its stored values."""

    name: str

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    @overload
    def __get__(self: _V, system: None, owner: type | None = None) -> _V: ...
    @overload
    def __get__(self, system: UnitSystem, owner: type | None = None) -> float: ...
    def __get__(self, system: UnitSystem | None, owner: type | None = None) -> Any:
        if system is None:
            return self
        return system._values[self.name]  # pyright: ignore[reportPrivateUsage]

    def compute(self, system: UnitSystem) -> float:
        raise NotImplementedError


class _PhysicalConstant(_StoredValue):
    """The physical constant of the same name, expressed in the system."""

    def __set_name__(self, owner: type, name: str) -> None:
        if name not in constants.CONSTANTS:
            raise NameError(f"unknown physical constant {name!r}")
        super().__set_name__(owner, name)

    def compute(self, system: UnitSystem) -> float:
        return system._physical_constant(self.name)  # pyright: ignore[reportPrivateUsage]


class _UnitFactor(_StoredValue):
    """One external unit expressed in the system."""

    def __init__(self, expression: str) -> None:
        self.expression = expression

    def compute(self, system: UnitSystem) -> float:
        return system.factor_from(self.expression)


def _resolve_unit(value: str | Unit, name: str, expected: Dimensions) -> Unit:
    unit = parse_unit(value) if isinstance(value, str) else value
    if type(unit) is not Unit:
        raise UnitSystemError(f"{name} must be a unit expression or Unit")
    if unit.dimensions != expected:
        raise UnitSystemError(
            f"{name} must have dimensions {expected}; got {unit.dimensions}"
        )
    return unit


@dataclass(frozen=True, slots=True, init=False)
class UnitSystem:
    """An immutable coherent working system defined by three mechanical units.

    Supply exactly three of ``length``, ``time``, ``energy``, and ``mass``.
    The missing unit is derived from ``E = M L^2 / T^2``. ``charge`` and
    ``temperature`` default to elementary charge and kelvin respectively.

    Conversion factors follow one convention: a named unit factor is one
    external unit expressed in this system, so values convert by multiplication.
    """

    length: Unit
    time: Unit
    energy: Unit
    mass: Unit
    charge: Unit
    temperature: Unit
    _scales: tuple[float, float, float, float, float] = field(repr=False)
    _values: MappingProxyType[str, float] = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        length: str | Unit | None = None,
        time: str | Unit | None = None,
        energy: str | Unit | None = None,
        mass: str | Unit | None = None,
        charge: str | Unit = "e",
        temperature: str | Unit = "K",
    ) -> None:
        supplied = {"length": length, "time": time, "energy": energy, "mass": mass}
        count = sum(value is not None for value in supplied.values())
        if count != 3:
            raise UnitSystemError(
                "exactly three of length, time, energy, and mass must be supplied; "
                f"got {count}"
            )

        scales: dict[str, float] = {}
        mechanical_dims = {"length": LENGTH, "time": TIME, "energy": ENERGY, "mass": MASS}
        resolved: dict[str, Unit] = {}
        for name, value in supplied.items():
            if value is not None:
                unit = _resolve_unit(value, name, mechanical_dims[name])
                resolved[name] = unit
                scales[name] = unit.scale

        if "mass" not in scales:
            scales["mass"] = scales["energy"] * scales["time"] ** 2 / scales["length"] ** 2
        elif "energy" not in scales:
            scales["energy"] = scales["mass"] * scales["length"] ** 2 / scales["time"] ** 2
        elif "length" not in scales:
            scales["length"] = math.sqrt(scales["energy"] * scales["time"] ** 2 / scales["mass"])
        else:
            scales["time"] = math.sqrt(scales["mass"] * scales["length"] ** 2 / scales["energy"])

        # Derived mechanical units are represented in SI and remain available
        # through the same explicit public attributes as supplied units.
        resolved.setdefault("length", Unit(scales["length"], LENGTH))
        resolved.setdefault("time", Unit(scales["time"], TIME))
        resolved.setdefault("energy", Unit(scales["energy"], ENERGY))
        resolved.setdefault("mass", Unit(scales["mass"], MASS))
        resolved["charge"] = _resolve_unit(charge, "charge", CHARGE)
        resolved["temperature"] = _resolve_unit(temperature, "temperature", TEMPERATURE)
        scales["charge"] = resolved["charge"].scale
        scales["temperature"] = resolved["temperature"].scale

        for name, unit in resolved.items():
            object.__setattr__(self, name, unit)
        object.__setattr__(
            self,
            "_scales",
            (scales["mass"], scales["length"], scales["time"], scales["charge"], scales["temperature"]),
        )

        # Physical constants and unit factors are computed once and stored as
        # plain floats, so reading them never re-enters the dimensional
        # machinery. Tracers such as torch.compile rely on this.
        values = {
            name: value.compute(self)
            for name, value in vars(type(self)).items()
            if isinstance(value, _StoredValue)
        }
        object.__setattr__(self, "_values", MappingProxyType(values))

    @staticmethod
    def _dims_scale(dimensions: Dimensions, scales: tuple[float, float, float, float, float]) -> float:
        exponents = (dimensions.mass, dimensions.length, dimensions.time, dimensions.charge, dimensions.temperature)
        # A plain loop rather than math.prod over a generator, which
        # torch.compile cannot trace.
        result = 1.0
        for scale, exponent in zip(scales, exponents):
            result *= scale ** float(exponent)
        return result

    def factor_from(self, unit: str | Unit) -> float:
        """Return one external unit expressed in this system as a float."""

        if isinstance(unit, str):
            resolved = parse_unit(unit)
        elif type(unit) is Unit:
            resolved = unit
        else:
            raise TypeError("unit must be a unit expression or Unit")
        return float(resolved.scale / self._dims_scale(resolved.dimensions, self._scales))

    def factor_to(self, unit: str | Unit) -> float:
        """Return the inverse of :meth:`factor_from`."""

        return float(1.0 / self.factor_from(unit))

    def to_internal(self, value: Any, unit: str | Unit) -> Any:
        """Convert an external value into this system by scalar multiplication."""

        return value * self.factor_from(unit)

    def from_internal(self, value: Any, unit: str | Unit) -> Any:
        """Convert a value from this system to the requested external unit."""

        return value * self.factor_to(unit)

    def _physical_constant(self, name: str) -> float:
        """Express an SI constant in this system, preserving dimensionless values."""

        constant = constants.CONSTANTS[name]
        return float(constant.value / self._dims_scale(constant.dimensions, self._scales))

    def constant(self, name: str) -> float:
        """Return a named physical constant expressed in this unit system.

        Constant names are case-sensitive and are listed by
        :func:`hartreez.constant_names`.
        """

        if type(name) is not str:
            raise TypeError("constant name must be a string")
        if name not in constants.CONSTANTS:
            raise ValueError(f"unknown physical constant {name!r}")
        return self._values[name]

    K_B = _PhysicalConstant()
    HBAR = _PhysicalConstant()
    PLANCK = _PhysicalConstant()
    C_LIGHT = _PhysicalConstant()
    ALPHA = _PhysicalConstant()
    M_E = _PhysicalConstant()
    M_P = _PhysicalConstant()
    E_CHARGE = _PhysicalConstant()
    N_A = _PhysicalConstant()
    EPSILON_0 = _PhysicalConstant()
    K_E = _PhysicalConstant()

    EV = _UnitFactor("eV")
    HARTREE = _UnitFactor("Hartree")
    HA = _UnitFactor("Ha")
    RY = _UnitFactor("Ry")
    ANGSTROM = _UnitFactor("angstrom")
    ANG = _UnitFactor("ang")
    BOHR = _UnitFactor("bohr")
    NM = _UnitFactor("nm")
    FS = _UnitFactor("fs")
    PS = _UnitFactor("ps")
    NS = _UnitFactor("ns")
    DA = _UnitFactor("Da")
    AMU = _UnitFactor("amu")
    KCALPERMOL = _UnitFactor("kcal/mol")
    KJPERMOL = _UnitFactor("kJ/mol")
    DEBYE = _UnitFactor("D")
    THZ = _UnitFactor("THz")
    CM1 = _UnitFactor("cm1")
    ATM = _UnitFactor("atm")
    BAR = _UnitFactor("bar")
    KBAR = _UnitFactor("kbar")
    GPA = _UnitFactor("GPa")
    PA = _UnitFactor("Pa")
    NEWTON = _UnitFactor("N")
    NN = _UnitFactor("nN")


au = UnitSystem(length="bohr", time="atomic_time", energy="Hartree", charge="e", temperature="K")
