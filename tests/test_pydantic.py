from __future__ import annotations

import subprocess
import sys
from copy import deepcopy
from typing import Annotated, Optional

import pytest
from pydantic import Field, ValidationError, field_validator

from hartreez import UnitDimension, UnitSystem, au, parse_unit
from hartreez.pydantic import UnitAwareModel


class InputModel(UnitAwareModel):
    timestep: Annotated[float, UnitDimension("ps")] = Field(gt=0.0)
    energy: Annotated[float, UnitDimension("eV")] = 0.0
    force: Annotated[float, UnitDimension("eV/angstrom")] = 0.0
    count: int = 1

    @field_validator("timestep")
    @classmethod
    def positive_internal_time(cls, value: float) -> float:
        assert value > 0
        return value


class ParticleInput(UnitAwareModel):
    mass: Annotated[float, UnitDimension("Da")]
    momentum: Annotated[float, UnitDimension("Da*angstrom/ps")]


class NestedInput(UnitAwareModel):
    timestep: Annotated[float, UnitDimension("fs")]
    particle: ParticleInput
    particles: list[ParticleInput]


class ThresholdParticleInput(UnitAwareModel):
    mass: Annotated[float, UnitDimension("Da")] = Field(gt=1.0)


class ThresholdNestedInput(UnitAwareModel):
    particle: ThresholdParticleInput


class OptionalInput(UnitAwareModel):
    temperature: Annotated[float, UnitDimension("K")] | None = None
    timestep: Optional[Annotated[float, UnitDimension("fs")]] = None


def test_unit_dimension_is_immutable_and_represents_only_dimensions() -> None:
    femtoseconds = UnitDimension("fs")
    picoseconds = UnitDimension("ps")
    momentum = UnitDimension("Da*angstrom/ps")

    assert femtoseconds.dimensions == picoseconds.dimensions
    assert momentum.dimensions == parse_unit("Da*angstrom/ps").dimensions
    with pytest.raises(AttributeError):
        femtoseconds.expression = "ps"  # type: ignore[misc]
    with pytest.raises(ValueError):
        UnitDimension("fortnight")


def test_pydantic_converts_scalars_in_the_selected_system_before_constraints() -> None:
    model = InputModel.model_validate(
        {"timestep[fs]": 100.0, "energy[eV]": 2.0, "force[eV/angstrom]": 1.0},
        context={"unit_system": UnitSystem(length="angstrom", time="ps", energy="eV")},
    )

    assert model.timestep == pytest.approx(0.1)
    assert model.energy == pytest.approx(2.0)
    assert model.force == pytest.approx(1.0)


def test_annotations_are_dimensional_and_do_not_choose_output_units() -> None:
    md = UnitSystem(length="angstrom", time="ps", energy="kcal/mol")

    model = InputModel.model_validate(
        {"timestep[fs]": 1.0, "energy[eV]": 1.0}, context={"unit_system": md}
    )

    assert model.energy == pytest.approx(md.factor_from("eV"))


def test_constraints_and_application_validators_receive_converted_values() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        InputModel.model_validate(
            {"timestep[fs]": -1.0}, context={"unit_system": au}
        )

    model = InputModel.model_validate(
        {"timestep[fs]": 1.0}, context={"unit_system": au}
    )
    assert model.timestep == pytest.approx(1.0 * au.factor_from("fs"))


def test_canonical_input_needs_no_context() -> None:
    model = InputModel.model_validate({"timestep": 0.25, "energy": 3.0})

    assert model.timestep == 0.25
    assert model.energy == 3.0


def test_annotated_input_requires_a_usable_system_context() -> None:
    with pytest.raises(ValidationError, match="requires a UnitSystem"):
        InputModel.model_validate({"timestep[fs]": 1.0})
    with pytest.raises(ValidationError, match="requires a UnitSystem"):
        InputModel.model_validate({"timestep[fs]": 1.0}, context={"unit_system": object()})


def test_dimensional_mismatch_and_normalization_collision_are_validation_errors() -> None:
    with pytest.raises(ValidationError, match="dimensional mismatch"):
        InputModel.model_validate(
            {"timestep[eV]": 1.0}, context={"unit_system": au}
        )
    with pytest.raises(ValidationError, match="normalized key collision"):
        InputModel.model_validate(
            {"timestep": 1.0, "timestep[fs]": 1.0}, context={"unit_system": au}
        )


def test_optional_annotated_fields_are_dimension_checked() -> None:
    md = UnitSystem(length="angstrom", time="ps", energy="kcal/mol")

    with pytest.raises(ValidationError, match="dimensional mismatch"):
        OptionalInput.model_validate(
            {"temperature[ps]": 3.0}, context={"unit_system": md}
        )


def test_optional_annotated_fields_convert_or_stay_none() -> None:
    md = UnitSystem(length="angstrom", time="ps", energy="kcal/mol")

    model = OptionalInput.model_validate(
        {"timestep[fs]": 100.0}, context={"unit_system": md}
    )
    assert model.timestep == pytest.approx(0.1)
    assert model.temperature is None

    with pytest.raises(ValidationError, match="dimensional mismatch"):
        OptionalInput.model_validate(
            {"timestep[K]": 1.0}, context={"unit_system": md}
        )


def test_conflicting_dimensions_are_rejected_when_the_model_is_defined() -> None:
    # The models are defined inside the test because defining them must fail.
    with pytest.raises(TypeError, match="conflicting UnitDimension"):

        class ConflictingMembers(UnitAwareModel):  # pyright: ignore[reportUnusedClass]
            value: Annotated[float, UnitDimension("fs")] | Annotated[
                int, UnitDimension("eV")
            ]

    with pytest.raises(TypeError, match="conflicting UnitDimension"):

        class ConflictingOuter(UnitAwareModel):  # pyright: ignore[reportUnusedClass]
            value: Annotated[
                Annotated[float, UnitDimension("fs")] | None, UnitDimension("K")
            ] = None


def test_unannotated_fields_keep_normal_pydantic_conversion_semantics() -> None:
    model = InputModel.model_validate(
        {"timestep[fs]": 1.0, "count": "4"}, context={"unit_system": au}
    )

    assert model.count == 4


def test_nested_models_and_model_lists_share_context_and_convert_their_own_fields() -> None:
    md = UnitSystem(length="angstrom", time="ps", energy="kcal/mol")
    payload = {
        "timestep[fs]": 100.0,
        "particle": {"mass[kg]": 1.0, "momentum[eV*ps/angstrom]": 2.0},
        "particles": [
            {"mass[Da]": 3.0, "momentum[Da*angstrom/ps]": 4.0},
            {"mass[kg]": 2.0, "momentum[eV*ps/angstrom]": 5.0},
        ],
    }
    original = deepcopy(payload)

    model = NestedInput.model_validate(payload, context={"unit_system": md})

    assert model.timestep == pytest.approx(0.1)
    assert model.particle.mass == pytest.approx(md.factor_from("kg"))
    assert model.particle.momentum == pytest.approx(
        2.0 * md.factor_from("eV*ps/angstrom")
    )
    assert model.particles[0].mass == pytest.approx(3.0 * md.factor_from("Da"))
    assert model.particles[0].momentum == pytest.approx(
        4.0 * md.factor_from("Da*angstrom/ps")
    )
    assert model.particles[1].mass == pytest.approx(2.0 * md.factor_from("kg"))
    assert model.particles[1].momentum == pytest.approx(
        5.0 * md.factor_from("eV*ps/angstrom")
    )
    assert payload == original


def test_nested_conversion_runs_before_field_constraints() -> None:
    # 0.001 Da is below the constraint as supplied, but above it after
    # conversion into atomic mass units.
    model = ThresholdNestedInput.model_validate(
        {"particle": {"mass[Da]": 0.001}}, context={"unit_system": au}
    )
    assert model.particle.mass == pytest.approx(0.001 * au.factor_from("Da"))
    assert model.particle.mass > 1.0


def test_nested_canonical_input_needs_no_context_but_raw_nested_input_does() -> None:
    canonical = NestedInput.model_validate(
        {
            "timestep": 1.0,
            "particle": {"mass": 2.0, "momentum": 3.0},
            "particles": [{"mass": 4.0, "momentum": 5.0}],
        }
    )
    assert canonical.particle.momentum == 3.0

    with pytest.raises(ValidationError) as exc_info:
        NestedInput.model_validate(
            {
                "timestep": 1.0,
                "particle": {"mass[Da]": 2.0, "momentum": 3.0},
                "particles": [],
            }
        )
    assert exc_info.value.errors()[0]["loc"] == ("particle",)
    assert "requires a UnitSystem" in str(exc_info.value)

    with pytest.raises(ValidationError) as list_exc_info:
        NestedInput.model_validate(
            {
                "timestep": 1.0,
                "particle": {"mass": 2.0, "momentum": 3.0},
                "particles": [{"mass": 4.0, "momentum[Da*angstrom/ps]": 5.0}],
            }
        )
    assert list_exc_info.value.errors()[0]["loc"] == ("particles", 0)


@pytest.mark.parametrize(
    ("bad_particle", "message"),
    [
        ({"mass[angstrom]": 1.0, "momentum": 2.0}, "dimensional mismatch"),
        (
            {"mass": 1.0, "mass[Da]": 2.0, "momentum": 3.0},
            "normalized key collision",
        ),
        ({"mass[Da]": True, "momentum": 2.0}, "unsupported value of type bool"),
    ],
)
def test_nested_conversion_errors_include_model_and_list_locations(
    bad_particle: dict[str, object], message: str
) -> None:
    parent_payload = {
        "timestep": 1.0,
        "particle": bad_particle,
        "particles": list[dict[str, object]](),
    }
    with pytest.raises(ValidationError) as nested_exc_info:
        NestedInput.model_validate(parent_payload, context={"unit_system": au})
    nested_error = nested_exc_info.value.errors()[0]
    assert nested_error["loc"] == ("particle",)
    assert message in nested_error["msg"]

    list_payload = {
        "timestep": 1.0,
        "particle": {"mass": 1.0, "momentum": 2.0},
        "particles": [bad_particle],
    }
    with pytest.raises(ValidationError) as list_exc_info:
        NestedInput.model_validate(list_payload, context={"unit_system": au})
    list_error = list_exc_info.value.errors()[0]
    assert list_error["loc"] == ("particles", 0)
    assert message in list_error["msg"]


def test_core_namespace_does_not_import_or_require_pydantic() -> None:
    script = """
import importlib.abc
import sys

class BlockPydantic(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pydantic' or fullname.startswith('pydantic.'):
            raise AssertionError('core import attempted to load Pydantic')
        return None

sys.meta_path.insert(0, BlockPydantic())
import hartreez
from hartreez import UnitDimension
assert UnitDimension('fs').dimensions == hartreez.parse_unit('ps').dimensions
assert 'pydantic' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)
