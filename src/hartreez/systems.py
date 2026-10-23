"""Coherent, immutable working unit systems."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from hartreez import constants
from hartreez.dimensions import Dimensions
from hartreez.errors import UnitSystemError
from hartreez.parser import parse_unit
from hartreez.units import CHARGE, ENERGY, LENGTH, MASS, TEMPERATURE, TIME, Unit

# Unit factor properties and the external unit each one expresses.
_UNIT_FACTORS: dict[str, str] = {
    "EV": "eV",
    "HARTREE": "Hartree",
    "HA": "Ha",
    "RY": "Ry",
    "ANGSTROM": "angstrom",
    "ANG": "ang",
    "BOHR": "bohr",
    "NM": "nm",
    "FS": "fs",
    "PS": "ps",
    "NS": "ns",
    "DA": "Da",
    "AMU": "amu",
    "KCALPERMOL": "kcal/mol",
    "KJPERMOL": "kJ/mol",
    "DEBYE": "D",
    "THZ": "THz",
    "CM1": "cm1",
    "ATM": "atm",
    "BAR": "bar",
    "KBAR": "kbar",
    "GPA": "GPa",
    "PA": "Pa",
    "NEWTON": "N",
    "NN": "nN",
}


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
        values = {name: self._physical_constant(name) for name in constants.CONSTANTS}
        values.update({name: self.factor_from(expression) for name, expression in _UNIT_FACTORS.items()})
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

    @property
    def K_B(self) -> float:
        return self._values["K_B"]

    @property
    def HBAR(self) -> float:
        return self._values["HBAR"]

    @property
    def PLANCK(self) -> float:
        return self._values["PLANCK"]

    @property
    def C_LIGHT(self) -> float:
        return self._values["C_LIGHT"]

    @property
    def ALPHA(self) -> float:
        return self._values["ALPHA"]

    @property
    def M_E(self) -> float:
        return self._values["M_E"]

    @property
    def M_P(self) -> float:
        return self._values["M_P"]

    @property
    def E_CHARGE(self) -> float:
        return self._values["E_CHARGE"]

    @property
    def N_A(self) -> float:
        return self._values["N_A"]

    @property
    def EPSILON_0(self) -> float:
        return self._values["EPSILON_0"]

    @property
    def K_E(self) -> float:
        return self._values["K_E"]

    @property
    def EV(self) -> float:
        return self._values["EV"]

    @property
    def HARTREE(self) -> float:
        return self._values["HARTREE"]

    @property
    def HA(self) -> float:
        return self._values["HA"]

    @property
    def RY(self) -> float:
        return self._values["RY"]

    @property
    def ANGSTROM(self) -> float:
        return self._values["ANGSTROM"]

    @property
    def ANG(self) -> float:
        return self._values["ANG"]

    @property
    def BOHR(self) -> float:
        return self._values["BOHR"]

    @property
    def NM(self) -> float:
        return self._values["NM"]

    @property
    def FS(self) -> float:
        return self._values["FS"]

    @property
    def PS(self) -> float:
        return self._values["PS"]

    @property
    def NS(self) -> float:
        return self._values["NS"]

    @property
    def DA(self) -> float:
        return self._values["DA"]

    @property
    def AMU(self) -> float:
        return self._values["AMU"]

    @property
    def KCALPERMOL(self) -> float:
        return self._values["KCALPERMOL"]

    @property
    def KJPERMOL(self) -> float:
        return self._values["KJPERMOL"]

    @property
    def DEBYE(self) -> float:
        return self._values["DEBYE"]

    @property
    def THZ(self) -> float:
        return self._values["THZ"]

    @property
    def CM1(self) -> float:
        return self._values["CM1"]

    @property
    def ATM(self) -> float:
        return self._values["ATM"]

    @property
    def BAR(self) -> float:
        return self._values["BAR"]

    @property
    def KBAR(self) -> float:
        return self._values["KBAR"]

    @property
    def GPA(self) -> float:
        return self._values["GPA"]

    @property
    def PA(self) -> float:
        return self._values["PA"]

    @property
    def NEWTON(self) -> float:
        return self._values["NEWTON"]

    @property
    def NN(self) -> float:
        return self._values["NN"]


au = UnitSystem(length="bohr", time="atomic_time", energy="Hartree", charge="e", temperature="K")
