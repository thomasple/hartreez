import pickle
import subprocess
import sys

import pytest

from hartreez import Dimensions, Unit, UnitSystem, UnitSystemError, au
from hartreez.systems import _UnitFactor  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("arguments", "derived_expression"),
    [
        ({"mass": "kg", "length": "m", "time": "s"}, "kg*m^2/s^2"),
        ({"length": "m", "time": "s", "energy": "J"}, "kg"),
        ({"mass": "kg", "time": "s", "energy": "J"}, "m"),
        ({"mass": "kg", "length": "m", "energy": "J"}, "s"),
    ],
)
def test_each_three_of_four_constructions(arguments: dict[str, str], derived_expression: str) -> None:
    system = UnitSystem(**arguments)  # type: ignore[arg-type]
    assert system.factor_from(derived_expression) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"length": "m"},
        {"length": "m", "time": "s"},
        {"length": "m", "time": "s", "energy": "J", "mass": "kg"},
    ],
)
def test_construction_requires_exactly_three_mechanical_units(arguments: dict[str, str]) -> None:
    with pytest.raises(UnitSystemError, match="exactly three"):
        UnitSystem(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "arguments",
    [
        {"length": "s", "time": "s", "energy": "J"},
        {"length": "m", "time": "kg", "energy": "J"},
        {"length": "m", "time": "s", "energy": "kg"},
        {"length": "m", "time": "s", "energy": "J", "charge": "K"},
        {"length": "m", "time": "s", "energy": "J", "temperature": "C"},
    ],
)
def test_constructor_rejects_incompatible_dimensions(arguments: dict[str, str]) -> None:
    with pytest.raises(UnitSystemError, match="must have dimensions"):
        UnitSystem(**arguments)  # type: ignore[arg-type]


def test_conversion_factors_round_trip_and_arbitrary_derived_units() -> None:
    system = UnitSystem(length="angstrom", time="ps", energy="kcal/mol")
    expression = "Da*angstrom/ps"
    factor = system.factor_from(expression)
    assert type(factor) is float
    assert system.factor_to(expression) == pytest.approx(1 / factor)
    converted = system.to_internal(2.5, expression)
    assert converted == pytest.approx(2.5 * factor)
    assert system.from_internal(converted, expression) == pytest.approx(2.5)
    assert system.factor_from("eV/angstrom") == pytest.approx(
        system.factor_from("eV") / system.factor_from("angstrom")
    )


def test_charge_and_temperature_defaults_and_overrides() -> None:
    default = UnitSystem(length="m", time="s", energy="J")
    millikelvin = Unit(1e-3, Dimensions(temperature=1))
    overridden = UnitSystem(length="m", time="s", energy="J", charge="C", temperature=millikelvin)
    assert default.factor_from("e") == 1.0
    assert default.factor_from("C") == pytest.approx(1 / 1.602176634e-19)
    assert default.factor_from("K") == 1.0
    assert overridden.factor_from("C") == 1.0
    assert overridden.factor_from("e") == pytest.approx(1.602176634e-19)
    assert overridden.factor_from(millikelvin) == 1.0
    assert overridden.factor_from("K") == pytest.approx(1000.0)


def test_atomic_system_conversion_direction_and_identities() -> None:
    assert au.factor_from("bohr") == pytest.approx(1.0)
    assert au.factor_from("atomic_time") == pytest.approx(1.0)
    assert au.factor_from("Hartree") == pytest.approx(1.0)
    assert au.factor_from("e") == pytest.approx(1.0)
    assert au.factor_from("K") == pytest.approx(1.0)
    assert au.mass.scale == pytest.approx(9.1093837139e-31)
    # Compare against CODATA 2022 table values; tolerances cover rounded inputs.
    assert 1.0 * au.EV == pytest.approx(1 / 27.211386245981, rel=2e-11)
    assert 1.0 * au.ANGSTROM == pytest.approx(1 / 0.529177210544, rel=2e-11)
    assert au.EV == au.factor_from("eV")
    assert au.ANGSTROM == au.factor_from("angstrom")
    assert au.HARTREE == au.factor_from("Hartree") == au.HA
    assert au.DA == au.factor_from("Da")
    assert au.KCALPERMOL == au.factor_from("kcal/mol")
    assert au.KJPERMOL == au.factor_from("kJ/mol")
    assert au.DEBYE == au.factor_from("D")
    assert au.THZ == au.factor_from("THz")
    assert au.CM1 == au.factor_from("cm1")
    for name, expression in (("ATM", "atm"), ("BAR", "bar"), ("GPA", "GPa"), ("PA", "Pa"), ("NEWTON", "N"), ("NN", "nN")):
        assert getattr(au, name) == au.factor_from(expression)


def test_unit_system_is_immutable() -> None:
    with pytest.raises((AttributeError, TypeError)):
        au.length = au.time  # type: ignore[misc]


def test_short_unit_names_in_working_system_and_expressions() -> None:
    md = UnitSystem(length="ang", time="ns", energy="meV")
    assert md.ANG == md.ANGSTROM == 1.0
    assert md.NS == 1.0
    assert md.NM == pytest.approx(10.0)
    assert md.RY == pytest.approx(md.HARTREE / 2)
    assert md.AMU == md.DA
    assert md.KBAR == pytest.approx(md.BAR * 1000)
    assert md.factor_from("eV/ang") == md.factor_from("eV/angstrom")
    assert au.ANG == au.ANGSTROM
    assert au.NS == pytest.approx(au.PS * 1000)
    assert au.NM == pytest.approx(au.ANGSTROM * 10)
    assert au.RY == pytest.approx(0.5)
    assert au.AMU == au.DA
    assert au.KBAR == pytest.approx(au.BAR * 1000)


def test_primary_au_import_loads_no_optional_frameworks() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from hartreez import au; "
            "assert au.EV > 0; "
            "assert not {'numpy', 'torch', 'jax', 'pydantic'} & sys.modules.keys()",
        ],
        check=True,
    )


def test_subclasses_inherit_constants_and_unit_factors() -> None:
    class MySystem(UnitSystem):
        pass

    system = MySystem(length="angstrom", time="ps", energy="kcal/mol")
    base = UnitSystem(length="angstrom", time="ps", energy="kcal/mol")
    assert system.K_B == base.K_B
    assert system.EV == base.EV
    assert system.constant("HBAR") == base.constant("HBAR")


def test_subclass_redeclaring_a_unit_factor_overrides_the_base() -> None:
    class MySystem(UnitSystem):
        EV = _UnitFactor("kcal/mol")

    system = MySystem(length="angstrom", time="ps", energy="kcal/mol")
    assert system.EV == 1.0


def test_unit_system_survives_a_pickle_round_trip() -> None:
    system = UnitSystem(length="angstrom", time="ps", energy="kcal/mol")
    restored = pickle.loads(pickle.dumps(system))
    assert restored == system
    assert restored.K_B == system.K_B
    assert restored.EV == system.EV
