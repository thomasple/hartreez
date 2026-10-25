from __future__ import annotations

import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, Generic, Optional, TypeVar, Union, cast

import pytest
from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    PlainValidator,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic.alias_generators import to_camel
from pydantic.dataclasses import dataclass as pydantic_dataclass
from typing_extensions import TypeAliasType, TypedDict

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


Length = Annotated[float, UnitDimension("angstrom")]


class SequenceInput(UnitAwareModel):
    positions: list[Length] = []
    cell: tuple[Length, Length, Length] = (1.0, 1.0, 1.0)
    shifts: tuple[Length, ...] = ()
    grid: list[list[Length]] = []
    cutoffs: list[Length] | None = None


class AliasedInput(UnitAwareModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    cutoff_radius: Length = 1.0
    skin: Length = Field(default=1.0, alias="rs")
    buffer: Length = Field(default=1.0, validation_alias="rb")


class StrictAliasInput(UnitAwareModel):
    skin: Length = Field(default=1.0, alias="rs")


class SharedAliasInput(UnitAwareModel):
    skin: Length = Field(default=1.0, validation_alias="r")
    cutoff: Annotated[float, UnitDimension("nm")] = Field(
        default=1.0, validation_alias="r"
    )


class ChoiceAliasInput(UnitAwareModel):
    cutoff: Length = Field(default=1.0, validation_alias=AliasChoices("rc", "cut"))


class ConstrainedAliasInput(UnitAwareModel):
    skin: Annotated[Length | None, Field(gt=0)] | None = None


# ``type Distance = ...`` on Python 3.12+ creates the same TypeAliasType.
Distance = TypeAliasType("Distance", Annotated[float, UnitDimension("angstrom")])


class TypeAliasInput(UnitAwareModel):
    cutoff: Distance = 1.0
    positions: list[Distance] = []


class ForwardRefInput(UnitAwareModel):
    cutoff: LaterLength


LaterLength = Annotated[float, UnitDimension("angstrom")]
ForwardRefInput.model_rebuild()


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
    with pytest.raises(TypeError, match="only Optional"):

        class ConflictingMembers(UnitAwareModel):  # pyright: ignore[reportUnusedClass]
            value: Annotated[float, UnitDimension("fs")] | Annotated[
                int, UnitDimension("eV")
            ]

    with pytest.raises(TypeError, match="conflicting UnitDimension"):

        class ConflictingOuter(UnitAwareModel):  # pyright: ignore[reportUnusedClass]
            value: Annotated[
                Annotated[float, UnitDimension("fs")] | None, UnitDimension("K")
            ] = None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("positions[ps]", [1.0, 2.0]),
        ("cell[ps]", (1.0, 2.0, 3.0)),
        ("shifts[ps]", (1.0,)),
        ("grid[ps]", [[1.0], [2.0]]),
        ("cutoffs[ps]", [1.0]),
    ],
)
def test_dimensions_declared_on_sequence_elements_are_checked(
    key: str, value: object
) -> None:
    with pytest.raises(ValidationError, match="dimensional mismatch"):
        SequenceInput.model_validate({key: value}, context={"unit_system": au})


def test_sequences_of_annotated_elements_convert_every_element() -> None:
    nm = UnitSystem(length="nm", time="ps", energy="kJ/mol")

    model = SequenceInput.model_validate(
        {"positions[angstrom]": [10.0, 20.0], "cell[angstrom]": (10.0, 20.0, 30.0)},
        context={"unit_system": nm},
    )

    assert model.positions == pytest.approx([1.0, 2.0])
    assert model.cell == pytest.approx((1.0, 2.0, 3.0))


def test_dimensions_nested_in_constrained_optional_aliases_are_checked() -> None:
    with pytest.raises(ValidationError, match="dimensional mismatch"):
        ConstrainedAliasInput.model_validate(
            {"skin[ps]": 1.0}, context={"unit_system": au}
        )


@pytest.mark.parametrize(
    "key", ["cutoffRadius[ps]", "cutoff_radius[ps]", "rs[ps]", "rb[ps]"]
)
def test_dimensions_are_checked_under_string_aliases(key: str) -> None:
    with pytest.raises(ValidationError, match="dimensional mismatch"):
        AliasedInput.model_validate({key: 1.0}, context={"unit_system": au})


def test_aliased_fields_convert_their_suffixed_input() -> None:
    nm = UnitSystem(length="nm", time="ps", energy="kJ/mol")

    model = AliasedInput.model_validate(
        {"cutoffRadius[angstrom]": 10.0, "rs[angstrom]": 20.0, "rb[angstrom]": 30.0},
        context={"unit_system": nm},
    )

    assert model.cutoff_radius == pytest.approx(1.0)
    assert model.skin == pytest.approx(2.0)
    assert model.buffer == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("model", "key"),
    [
        (InputModel, "count[ps]"),
        (InputModel, "undeclared[fs]"),
    ],
)
def test_suffixed_keys_without_a_declared_dimension_are_rejected(
    model: type[UnitAwareModel], key: str
) -> None:
    with pytest.raises(ValidationError, match="no reference unit declared"):
        model.model_validate(
            {"timestep": 1.0, key: 1.0}, context={"unit_system": au}
        )


@pytest.mark.parametrize("key", ["rc", "cut"])
def test_every_alias_choice_converts_its_suffixed_input(key: str) -> None:
    nm = UnitSystem(length="nm", time="ps", energy="kJ/mol")

    model = ChoiceAliasInput.model_validate(
        {f"{key}[angstrom]": 10.0}, context={"unit_system": nm}
    )
    assert model.cutoff == pytest.approx(1.0)

    with pytest.raises(ValidationError, match="dimensional mismatch"):
        ChoiceAliasInput.model_validate({f"{key}[ps]": 1.0}, context={"unit_system": nm})


def test_field_names_follow_pydantic_alias_configuration() -> None:
    # Without populate_by_name, Pydantic ignores the field name of an aliased
    # field, so its suffixed key must not be converted and then dropped.
    with pytest.raises(ValidationError, match="no reference unit declared"):
        StrictAliasInput.model_validate(
            {"skin[angstrom]": 1.0}, context={"unit_system": au}
        )


def test_input_keys_shared_with_one_dimension_are_accepted() -> None:
    nm = UnitSystem(length="nm", time="ps", energy="kJ/mol")

    model = SharedAliasInput.model_validate(
        {"r[angstrom]": 10.0}, context={"unit_system": nm}
    )

    assert model.skin == pytest.approx(1.0)
    assert model.cutoff == pytest.approx(1.0)


def test_input_keys_shared_with_conflicting_dimensions_are_rejected() -> None:
    # The models are defined inside the test because defining them must fail.
    with pytest.raises(TypeError, match="conflicting UnitDimension"):

        class SharedAlias(UnitAwareModel):  # pyright: ignore[reportUnusedClass]
            length: Length = Field(default=1.0, validation_alias="v")
            time: Annotated[float, UnitDimension("fs")] = Field(
                default=1.0, validation_alias="v"
            )

    with pytest.raises(TypeError, match="conflicting UnitDimension"):

        class AliasOntoName(UnitAwareModel):  # pyright: ignore[reportUnusedClass]
            model_config = ConfigDict(populate_by_name=True)

            length: Length = 1.0
            time: Annotated[float, UnitDimension("fs")] = Field(
                default=1.0, validation_alias="length"
            )


def test_dimensions_declared_through_type_aliases_are_found() -> None:
    nm = UnitSystem(length="nm", time="ps", energy="kJ/mol")

    model = TypeAliasInput.model_validate(
        {"cutoff[angstrom]": 10.0, "positions[angstrom]": [20.0]},
        context={"unit_system": nm},
    )
    assert model.cutoff == pytest.approx(1.0)
    assert model.positions == pytest.approx([2.0])

    with pytest.raises(ValidationError, match="dimensional mismatch"):
        TypeAliasInput.model_validate({"cutoff[ps]": 1.0}, context={"unit_system": nm})


class PlainSettings(BaseModel):
    label: str = ""


class PlainParticle(BaseModel):
    mass: Annotated[float, UnitDimension("Da")] = 1.0


def test_plain_models_without_dimensions_can_be_nested() -> None:
    class WithSettings(UnitAwareModel):
        settings: PlainSettings

    model = WithSettings.model_validate({"settings": {"label": "md"}})
    assert model.settings.label == "md"


@dataclass
class DataclassParticle:
    mass: Annotated[float, UnitDimension("Da")] = 1.0


@pydantic_dataclass
class PydanticDataclassParticle:
    mass: Annotated[float, UnitDimension("Da")] = 1.0


class TypedDictParticle(TypedDict, total=False):
    mass: Annotated[float, UnitDimension("Da")]


class PlainHolder(BaseModel):
    particle: PlainParticle = PlainParticle()


@dataclass
class DataclassHolder:
    particles: list[PlainParticle]


Item = TypeVar("Item")


@dataclass
class DataclassBox(Generic[Item]):
    item: Item
    mass: Annotated[float, UnitDimension("Da")] = 1.0


@dataclass
class DataclassSlot(Generic[Item]):
    item: Item


class TypedDictBox(TypedDict, Generic[Item], total=False):
    item: Item
    mass: Annotated[float, UnitDimension("Da")]


class TypedDictSlot(TypedDict, Generic[Item]):
    item: Item


class PlainTree(BaseModel):
    label: str = ""
    children: list[PlainTree] = []


@pytest.mark.parametrize(
    ("annotation", "offender"),
    [
        (PlainParticle, "PlainParticle"),
        (list[PlainParticle], "PlainParticle"),
        (PlainParticle | None, "PlainParticle"),
        (dict[str, PlainParticle], "PlainParticle"),
        (DataclassParticle, "DataclassParticle"),
        (PydanticDataclassParticle, "PydanticDataclassParticle"),
        (TypedDictParticle, "TypedDictParticle"),
        (PlainHolder, "PlainParticle"),
        (DataclassHolder, "PlainParticle"),
        (DataclassBox[int], "DataclassBox"),
        (DataclassSlot[Length], "DataclassSlot"),
        (TypedDictBox[int], "TypedDictBox"),
        (TypedDictSlot[Length], "TypedDictSlot"),
    ],
)
def test_plain_types_declaring_dimensions_are_rejected_when_nested(
    annotation: object, offender: str
) -> None:
    # A plain model, dataclass or TypedDict would drop its suffixed keys as
    # unknown keys, leaving defaults behind, at any nesting depth.
    with pytest.raises(TypeError, match=f"{offender}.*UnitAwareModel"):
        type("Outer", (UnitAwareModel,), {"__annotations__": {"particle": annotation}})


def test_generic_plain_types_without_dimensions_can_be_nested() -> None:
    class WithSlots(UnitAwareModel):
        slot: DataclassSlot[int]
        entry: TypedDictSlot[str]

    model = WithSlots.model_validate({"slot": {"item": 1}, "entry": {"item": "a"}})
    assert model.slot.item == 1


def test_recursive_plain_models_without_dimensions_can_be_nested() -> None:
    class WithTree(UnitAwareModel):
        tree: PlainTree

    model = WithTree.model_validate({"tree": {"children": [{"label": "leaf"}]}})
    assert model.tree.children[0].label == "leaf"


def test_input_keys_shared_with_undimensioned_fields_are_rejected() -> None:
    # The models are defined inside the test because defining them must fail.
    with pytest.raises(TypeError, match="'length' and 'raw'"):

        class SharedAlias(UnitAwareModel):  # pyright: ignore[reportUnusedClass]
            length: Length = Field(default=1.0, validation_alias="v")
            raw: float = Field(default=1.0, validation_alias="v")

    with pytest.raises(TypeError, match="'length' and 'raw'"):

        class AliasOntoName(UnitAwareModel):  # pyright: ignore[reportUnusedClass]
            model_config = ConfigDict(populate_by_name=True)

            length: Length = 1.0
            raw: float = Field(default=1.0, validation_alias="length")

    with pytest.raises(TypeError, match="'length' and 'raw'"):

        class SharedChoice(UnitAwareModel):  # pyright: ignore[reportUnusedClass]
            length: Length = Field(default=1.0, validation_alias="v")
            raw: float = Field(default=1.0, validation_alias=AliasChoices("w", "v"))

    with pytest.raises(TypeError, match="'length' and 'raw'"):

        class SharedPath(UnitAwareModel):  # pyright: ignore[reportUnusedClass]
            length: Length = Field(default=1.0, validation_alias="v")
            raw: float = Field(default=1.0, validation_alias=AliasPath("v", 0))

    with pytest.raises(TypeError, match="'raw' and 'length'"):

        class UndimensionedFirst(UnitAwareModel):  # pyright: ignore[reportUnusedClass]
            raw: float = Field(default=1.0, validation_alias="v")
            length: Length = Field(default=1.0, validation_alias="v")


@pytest.mark.parametrize(
    "alias", [AliasPath("outer", "x"), AliasChoices("x", AliasPath("outer", "x"))]
)
def test_alias_paths_on_dimensioned_fields_are_rejected_when_the_model_is_defined(
    alias: AliasPath | AliasChoices,
) -> None:
    # A suffix inside a path would never reach the model's own mapping level.
    with pytest.raises(TypeError, match="field 'x'.*AliasPath"):
        type(
            "PathInput",
            (UnitAwareModel,),
            {
                "__annotations__": {"x": Length},
                "x": Field(default=1.0, validation_alias=alias),
            },
        )


def test_input_keys_shared_only_by_undimensioned_fields_are_allowed() -> None:
    class SharedRaw(UnitAwareModel):
        first: float = Field(default=0.0, validation_alias="v")
        second: float = Field(default=0.0, validation_alias="v")

    model = SharedRaw.model_validate({"v": 2.0})
    assert (model.first, model.second) == (2.0, 2.0)


def test_dimensions_resolved_by_model_rebuild_are_checked() -> None:
    with pytest.raises(ValidationError, match="dimensional mismatch"):
        ForwardRefInput.model_validate({"cutoff[ps]": 1.0}, context={"unit_system": au})


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


class TaggedSettings(BaseModel):
    model_config = ConfigDict(extra="allow")

    tags: dict[str, float] = {}


class ContainerInput(UnitAwareModel):
    model_config = ConfigDict(extra="allow")

    masses: dict[str, float] = {}
    tables: list[dict[str, float]] = []
    settings: TaggedSettings = TaggedSettings()
    particles: dict[str, ParticleInput] = {}


@pytest.mark.parametrize(
    ("payload", "location"),
    [
        ({"masses": {"H[Da]": 1.0}}, r"\$\.masses"),
        ({"tables": [{}, {"H[Da]": 1.0}]}, r"\$\.tables\[1\]"),
        ({"settings": {"tags": {"H[Da]": 1.0}}}, r"\$\.settings\.tags"),
        ({"settings": {"H[Da]": 1.0}}, r"\$\.settings"),
        ({"extra": {"H[Da]": 1.0}}, r"\$\.extra"),
    ],
)
def test_suffixed_keys_outside_unit_aware_models_are_rejected(
    payload: dict[str, object], location: str
) -> None:
    # Only a UnitAwareModel converts its keys; elsewhere the suffix would stay
    # part of the key and its value would never be converted.
    with pytest.raises(ValidationError, match=rf"'H\[Da\]' at {location}"):
        ContainerInput.model_validate(payload, context={"unit_system": au})


def test_suffixed_keys_inside_nested_unit_aware_models_still_convert() -> None:
    model = ContainerInput.model_validate(
        {"particles": {"a": {"mass[Da]": 1.0, "momentum": 0.0}}},
        context={"unit_system": au},
    )
    assert model.particles["a"].mass == pytest.approx(au.factor_from("Da"))


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


class AnyPayloadInput(UnitAwareModel):
    payload: Any = None


def test_cyclic_values_without_suffixes_are_accepted() -> None:
    cycle: list[Any] = []
    cycle.append(cycle)

    model = AnyPayloadInput.model_validate({"payload": cycle})
    assert model.payload is cycle


class PlainMass(BaseModel):
    mass: float = 1.0


def _outer(annotation: object, **namespace: object) -> type[UnitAwareModel]:
    return type(
        "Outer", (UnitAwareModel,), {"__annotations__": {"value": annotation}, **namespace}
    )


@pytest.mark.parametrize(
    ("model", "payload", "location"),
    [
        # Plain nested types without dimensions drop unknown keys by default.
        (_outer(PlainMass), {"value": {"mass[Da]": 12.0}}, r"\$\.value"),
        (_outer(Any), {"value": {"mass[Da]": 12.0}}, r"\$\.value"),
        # A path alias reads below the model's own level.
        (
            _outer(float, value=Field(0.0, validation_alias=AliasPath("w", "v"))),
            {"w": {"v[Da]": 12.0}},
            r"\$\.w",
        ),
        # The compiled schema cannot see a model validated by custom code.
        (
            _outer(Annotated[Any, PlainValidator(PlainParticle.model_validate)]),
            {"value": {"mass[Da]": 12.0}},
            r"\$\.value",
        ),
        # Pydantic prefers the dict member, which keeps the suffixed key.
        (
            _outer(Union[ParticleInput, dict[str, float]]),
            {"value": {"mass[Da]": 12.0}},
            r"\$\.value",
        ),
    ],
)
def test_suffixed_keys_not_converted_by_a_unit_aware_model_are_rejected(
    model: type[UnitAwareModel], payload: dict[str, object], location: str
) -> None:
    with pytest.raises(ValidationError, match=rf"'\w+\[Da\]' at {location} has a unit suffix"):
        model.model_validate(payload, context={"unit_system": au})


def test_a_dict_object_shared_by_two_nested_models_converts_in_both() -> None:
    class Particles(UnitAwareModel):
        particles: list[ParticleInput]

    shared = {"mass[Da]": 1.0, "momentum": 0.0}
    model = Particles.model_validate(
        {"particles": [shared, shared]}, context={"unit_system": au}
    )
    assert [p.mass for p in model.particles] == pytest.approx([au.factor_from("Da")] * 2)


class CopiedParticle(UnitAwareModel):
    particle: ParticleInput

    @field_validator("particle", mode="before")
    @classmethod
    def copy_particle(cls, value: Any) -> Any:
        return dict(value)


class DeepCopiedParticles(UnitAwareModel):
    particles: list[ParticleInput]

    @model_validator(mode="before")
    @classmethod
    def copy_input(cls, data: Any) -> Any:
        return deepcopy(data)


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (CopiedParticle, {"particle": {"mass[Da]": 1.0, "momentum": 0.0}}),
        (
            DeepCopiedParticles,
            {"particles": [{"mass[Da]": 1.0, "momentum": 0.0}, {"mass[Da]": 1.0, "momentum": 0.0}]},
        ),
    ],
)
def test_validators_may_copy_nested_input(
    model: type[UnitAwareModel], payload: dict[str, object]
) -> None:
    # Nested models are matched to their input by identity, then by equal
    # contents, so a copy of a nested input mapping still counts.
    model.model_validate(payload, context={"unit_system": au})


class NamedParticle(UnitAwareModel):
    name: str = ""
    mass: Annotated[float, UnitDimension("Da")] = 1.0


class Species(UnitAwareModel):
    particles: list[NamedParticle]

    @field_validator("particles", mode="before")
    @classmethod
    def from_mapping(cls, value: Any) -> Any:
        # Shorthand: species keyed by name.
        if isinstance(value, dict):
            entries = cast(dict[str, dict[str, Any]], value)
            return [{"name": name, **entry} for name, entry in entries.items()]
        return value


class WithDefaults(UnitAwareModel):
    particles: list[NamedParticle]

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, data: Any) -> Any:
        defaults = data.pop("defaults", {})
        return {**data, "particles": [{**defaults, **p} for p in data["particles"]]}


class Commented(UnitAwareModel):
    particle: NamedParticle

    @field_validator("particle", mode="before")
    @classmethod
    def drop_comments(cls, value: Any) -> Any:
        return {k: v for k, v in value.items() if not k.startswith("_")}


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (Species, {"particles": {"H": {"mass[Da]": 1.0}, "O": {"mass[Da]": 16.0}}}),
        (WithDefaults, {"defaults": {"name": "X"}, "particles": [{"mass[Da]": 1.0}]}),
        (Commented, {"particle": {"_comment": "hydrogen", "mass[Da]": 1.0}}),
    ],
)
def test_validators_may_add_or_remove_unsuffixed_keys_of_nested_input(
    model: type[UnitAwareModel], payload: dict[str, object]
) -> None:
    # A changed copy is still matched while every suffixed entry of its
    # input is unchanged.
    model.model_validate(payload, context={"unit_system": au})


def test_validators_that_rename_suffixed_keys_of_nested_input_are_rejected() -> None:
    class Migrating(UnitAwareModel):
        particle: NamedParticle

        @field_validator("particle", mode="before")
        @classmethod
        def rename_legacy_keys(cls, value: Any) -> Any:
            return {("mass[Da]" if k == "m[Da]" else k): v for k, v in value.items()}

    with pytest.raises(ValidationError, match=r"'m\[Da\]' at \$\.particle"):
        Migrating.model_validate({"particle": {"m[Da]": 1.0}}, context={"unit_system": au})


def test_suffixed_defaults_merged_by_a_validator_convert() -> None:
    model = WithDefaults.model_validate(
        {"defaults": {"mass[Da]": 1.0}, "particles": [{"name": "H"}, {"name": "D"}]},
        context={"unit_system": au},
    )
    assert [p.mass for p in model.particles] == pytest.approx([au.factor_from("Da")] * 2)


def test_suffixed_defaults_that_no_nested_model_receives_are_rejected() -> None:
    with pytest.raises(ValidationError, match=r"'mass\[Da\]' at \$\.defaults"):
        WithDefaults.model_validate(
            {"defaults": {"mass[Da]": 1.0}, "particles": []}, context={"unit_system": au}
        )


def test_a_copy_converts_only_one_of_two_equal_inputs() -> None:
    class Pair(UnitAwareModel):
        particle: ParticleInput
        dropped: PlainMass

        @field_validator("particle", mode="before")
        @classmethod
        def copy_particle(cls, value: Any) -> Any:
            return dict(value)

    same = {"mass[Da]": 1.0, "momentum": 0.0}
    with pytest.raises(ValidationError, match=r"'mass\[Da\]' at \$\.dropped"):
        Pair.model_validate(
            {"particle": dict(same), "dropped": dict(same)}, context={"unit_system": au}
        )


def test_a_nested_models_own_validator_may_rename_suffixed_keys() -> None:
    class LegacyParticle(UnitAwareModel):
        mass: Annotated[float, UnitDimension("Da")] = 1.0

        @model_validator(mode="before")
        @classmethod
        def rename_legacy_keys(cls, data: Any) -> Any:
            return {("mass[Da]" if k == "m[Da]" else k): v for k, v in data.items()}

    class Holder(UnitAwareModel):
        particle: LegacyParticle

    model = Holder.model_validate({"particle": {"m[Da]": 1.0}}, context={"unit_system": au})
    assert model.particle.mass == pytest.approx(au.factor_from("Da"))


class SharedInput(UnitAwareModel):
    particle: ParticleInput
    raw: Any = None
    plain: Optional[PlainMass] = None


@pytest.mark.parametrize(
    ("second", "message"),
    [
        # Kept unconverted in an Any field, and reported where it is kept.
        ("raw", r"'mass\[Da\]' at \$\.raw has a unit suffix"),
        # Dropped as unknown keys by a plain model.
        ("plain", r"'mass\[Da\]' at \$\.particle .*appears 2 times"),
    ],
)
def test_a_mapping_used_twice_must_be_converted_at_each_place(
    second: str, message: str
) -> None:
    shared = {"mass[Da]": 1.0, "momentum": 0.0}
    with pytest.raises(ValidationError, match=message):
        SharedInput.model_validate(
            {"particle": shared, second: shared}, context={"unit_system": au}
        )


def test_suffixed_mappings_inside_any_sequence_are_found() -> None:
    from collections import deque

    with pytest.raises(ValidationError, match=r"'mass\[Da\]' at \$\.value\[0\]"):
        _outer(Any).model_validate(
            {"value": deque([{"mass[Da]": 1.0}])}, context={"unit_system": au}
        )


def test_heavily_shared_input_is_walked_once_per_object() -> None:
    # 40 levels of a list holding its child twice: 2**40 paths, 40 objects.
    value: Any = {"mass": 1.0}
    for _ in range(40):
        value = [value, value]
    _outer(Any).model_validate({"value": value}, context={"unit_system": au})


def test_validators_that_change_a_suffixed_value_type_are_rejected() -> None:
    # True == 1.0 in Python, but the value changed, so the copy matches no input.
    class Retyping(UnitAwareModel):
        particle: ParticleInput

        @field_validator("particle", mode="before")
        @classmethod
        def to_float(cls, value: Any) -> Any:
            return {k: (float(v) if isinstance(v, bool) else v) for k, v in value.items()}

    with pytest.raises(ValidationError, match=r"'mass\[Da\]' at \$\.particle"):
        Retyping.model_validate(
            {"particle": {"mass[Da]": True, "momentum": 0.0}}, context={"unit_system": au}
        )


def test_duplicated_nested_input_does_not_cover_a_kept_occurrence() -> None:
    class Duplicating(UnitAwareModel):
        particles: list[ParticleInput]
        raw: Any = None

        @field_validator("particles", mode="before")
        @classmethod
        def duplicate(cls, value: Any) -> Any:
            return [value[0], value[0]]

    shared = {"mass[Da]": 1.0, "momentum": 0.0}
    with pytest.raises(ValidationError, match=r"'mass\[Da\]' at \$\.raw"):
        Duplicating.model_validate(
            {"particles": [shared], "raw": shared}, context={"unit_system": au}
        )


def test_validators_that_change_set_member_types_are_rejected() -> None:
    class Flags(UnitAwareModel):
        values: set[Annotated[float, UnitDimension("Da")]] = set()

    class Holder(UnitAwareModel):
        flags: Flags

        @field_validator("flags", mode="before")
        @classmethod
        def to_floats(cls, value: Any) -> Any:
            return {k: {float(x) for x in v} for k, v in value.items()}

    with pytest.raises(ValidationError, match=r"'values\[Da\]' at \$\.flags"):
        Holder.model_validate({"flags": {"values[Da]": {True}}}, context={"unit_system": au})
