from typing import Any

import pytest

from hartreez import UnitSystem, constant_names


def _public_property_names() -> list[str]:
    return [name for name, value in vars(UnitSystem).items() if isinstance(value, property)]


def test_constants_and_unit_factors_are_stored_not_recomputed(monkeypatch: pytest.MonkeyPatch) -> None:
    system = UnitSystem(length="angstrom", time="ps", energy="kcal/mol")
    properties = {name: getattr(system, name) for name in _public_property_names()}
    constants = {name: system.constant(name) for name in constant_names()}

    def fail(*args: object, **kwargs: object) -> float:
        raise AssertionError("value was recomputed on access")

    monkeypatch.setattr(UnitSystem, "_dims_scale", staticmethod(fail))
    monkeypatch.setattr(UnitSystem, "_physical_constant", fail)
    monkeypatch.setattr(UnitSystem, "factor_from", fail)

    assert {name: getattr(system, name) for name in properties} == properties
    assert {name: system.constant(name) for name in constants} == constants
    assert all(type(value) is float for value in properties.values())


def test_stored_values_do_not_affect_equality_or_hashing() -> None:
    first = UnitSystem(length="angstrom", time="ps", energy="kcal/mol")
    second = UnitSystem(length="angstrom", time="ps", energy="kcal/mol")
    assert first == second
    assert hash(first) == hash(second)


def test_constant_rejects_unit_factor_names() -> None:
    system = UnitSystem(length="angstrom", time="ps", energy="kcal/mol")
    with pytest.raises(ValueError, match="unknown physical constant"):
        system.constant("EV")


def test_torch_compile_traces_constants_and_conversions() -> None:
    torch = pytest.importorskip("torch")
    system = UnitSystem(length="angstrom", time="ps", energy="kcal/mol")

    def f(x: Any) -> Any:
        y = x * system.K_B * system.FS * system.AMU * system.EV * system.BOHR
        y = y * system.constant("HBAR") * system.factor_from("eV/angstrom")
        return system.from_internal(system.to_internal(y, "eV/angstrom"), "eV")

    compiled = torch.compile(f, fullgraph=True, backend="aot_eager")
    x = torch.ones(2, dtype=torch.float64)
    torch.testing.assert_close(compiled(x), f(x))
