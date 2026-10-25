from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Annotated, TypeVar

import pytest
from pydantic import BaseModel, ValidationError
from typing_extensions import TypeAliasType

from hartreez import UnitDimension, UnitSystem, au
from hartreez.pydantic import UnitAwareModel


Length = Annotated[float, UnitDimension("angstrom")]
Time = Annotated[float, UnitDimension("ps")]
Distance = TypeAliasType("Distance", Length)
T = TypeVar("T")
# ``type Vector[T] = list[T]`` on Python 3.12+ creates the same generic alias.
Vector = TypeAliasType("Vector", list[T], type_params=(T,))


class ProjectedInput(UnitAwareModel):
    scalar: Length = 1.0
    optional: Length | None = None
    positions: list[list[Length]] = []
    sequence: Sequence[Length] = ()
    fixed: tuple[Length, Length] = (1.0, 1.0)
    variable: tuple[Distance, ...] = ()


def test_supported_annotations_project_one_dimension_onto_the_field() -> None:
    nm = UnitSystem(length="nm", time="ps", energy="kJ/mol")

    model = ProjectedInput.model_validate(
        {
            "scalar[angstrom]": 10.0,
            "optional[angstrom]": 20.0,
            "positions[angstrom]": [[10.0, 20.0], [30.0, 40.0]],
            "sequence[angstrom]": [50.0, 60.0],
            "fixed[angstrom]": [70.0, 80.0],
            "variable[angstrom]": [90.0, 100.0],
        },
        context={"unit_system": nm},
    )

    assert model.scalar == pytest.approx(1.0)
    assert model.optional == pytest.approx(2.0)
    assert model.positions[0] == pytest.approx([1.0, 2.0])
    assert model.positions[1] == pytest.approx([3.0, 4.0])
    assert list(model.sequence) == pytest.approx([5.0, 6.0])
    assert model.fixed == pytest.approx((7.0, 8.0))
    assert model.variable == pytest.approx((9.0, 10.0))


def test_projected_dimension_is_checked_before_pydantic_validation() -> None:
    with pytest.raises(ValidationError, match="dimensional mismatch"):
        ProjectedInput.model_validate(
            {"positions[ps]": [[1.0]]}, context={"unit_system": au}
        )


@pytest.mark.parametrize(
    "annotation",
    [
        dict[str, Length],
        deque[Length],
        Length | str,
        tuple[Length, int],
        Vector[Length],
    ],
)
def test_dimension_inside_unsupported_shape_fails_at_model_definition(
    annotation: object,
) -> None:
    with pytest.raises(TypeError, match="field 'value'"):
        type(
            "Unsupported",
            (UnitAwareModel,),
            {"__annotations__": {"value": annotation}},
        )


def test_conflicting_dimensions_in_homogeneous_tuple_fail_at_model_definition() -> None:
    with pytest.raises(TypeError, match="conflicting UnitDimension"):
        type(
            "MixedDimensions",
            (UnitAwareModel,),
            {"__annotations__": {"value": tuple[Length, Time]}},
        )


def test_same_dimension_can_be_repeated_at_multiple_annotation_levels() -> None:
    class Repeated(UnitAwareModel):
        value: Annotated[Length, UnitDimension("nm")]

    nm = UnitSystem(length="nm", time="ps", energy="kJ/mol")
    model = Repeated.model_validate(
        {"value[angstrom]": 10.0}, context={"unit_system": nm}
    )
    assert model.value == pytest.approx(1.0)


class PlainParticle(BaseModel):
    mass: Annotated[float, UnitDimension("Da")] = 1.0


def test_plain_models_with_dimensions_are_found_through_generic_aliases() -> None:
    with pytest.raises(TypeError, match="PlainParticle.*UnitAwareModel"):
        type(
            "Outer",
            (UnitAwareModel,),
            {"__annotations__": {"particles": Vector[PlainParticle]}},
        )


class SetInput(UnitAwareModel):
    values: set[Length] = set()
    frozen: frozenset[Length] = frozenset()


def test_set_shapes_convert_native_set_input() -> None:
    nm = UnitSystem(length="nm", time="ps", energy="kJ/mol")

    model = SetInput.model_validate(
        {"values[angstrom]": {10.0, 20.0}, "frozen[angstrom]": frozenset({30.0})},
        context={"unit_system": nm},
    )

    assert sorted(model.values) == pytest.approx([1.0, 2.0])
    assert sorted(model.frozen) == pytest.approx([3.0])


@pytest.mark.parametrize("value", [deque([1.0]), range(3)])
def test_other_iterables_under_a_suffix_are_rejected(value: object) -> None:
    # Conversion runs before Pydantic, on lists, tuples, sets and frozensets.
    with pytest.raises(ValidationError, match="unsupported value of type"):
        ProjectedInput.model_validate(
            {"sequence[angstrom]": value}, context={"unit_system": au}
        )
