from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pytest

from hartreez import MappingConversionError, ReferenceUnitsError, UnitSystem, au, convert_mapping


def test_converts_scalars_and_nested_lists_and_tuples() -> None:
    result = convert_mapping(
        {"dt[fs]": 2.0, "cell[angstrom]": [[1.0, 0.0], [0.0, 2.0]], "pair[eV]": (1, 2)},
        au,
    )

    assert result["dt"] == pytest.approx(2.0 * au.FS)
    assert result["cell"] == [
        [au.ANGSTROM, 0.0],
        [0.0, 2.0 * au.ANGSTROM],
    ]
    assert result["pair"] == pytest.approx((au.EV, 2.0 * au.EV))
    assert isinstance(result["pair"], tuple)


def test_converts_sets_and_frozensets_keeping_their_type() -> None:
    result = convert_mapping({"r[angstrom]": {1.0, 2.0}, "e[eV]": frozenset({3.0})}, au)

    assert isinstance(result["r"], set)
    assert sorted(cast(set[float], result["r"])) == pytest.approx([au.ANGSTROM, 2.0 * au.ANGSTROM])
    assert isinstance(result["e"], frozenset)
    assert sorted(cast(frozenset[float], result["e"])) == pytest.approx([3.0 * au.EV])


def test_recurses_through_nested_mappings_and_sequences() -> None:
    source = {"simulation": {"dt[fs]": 3.0}, "replicas": [{"r[angstrom]": 1.0}]}
    result = convert_mapping(source, au)

    assert result == {
        "simulation": {"dt": 3.0 * au.FS},
        "replicas": [{"r": au.ANGSTROM}],
    }
    assert source == {"simulation": {"dt[fs]": 3.0}, "replicas": [{"r[angstrom]": 1.0}]}


def test_unannotated_values_and_non_string_keys_are_preserved() -> None:
    marker = object()
    source = {1: marker, (2, 3): "literal", "label": "dt[fs]", "nested": {4: marker}}

    result = convert_mapping(source, au)

    assert result[1] is marker
    assert result[(2, 3)] == "literal"
    assert result["label"] == "dt[fs]"
    assert result["nested"][4] is marker


def test_does_not_mutate_input_and_returns_fresh_nested_mappings() -> None:
    child = {"dt[fs]": [1.0, 2.0]}
    source = {"child": child}

    result = convert_mapping(source, au)

    assert result is not source
    assert result["child"] is not child
    assert child == {"dt[fs]": [1.0, 2.0]}


class ArrayLike:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def __mul__(self, factor: float) -> ArrayLike:
        return ArrayLike([value * factor for value in self.values])


def test_scales_array_like_values_through_multiplication_duck_typing() -> None:
    result = convert_mapping({"cell[angstrom]": ArrayLike([1.0, 2.0])}, au)
    scaled = result["cell"]

    assert isinstance(scaled, ArrayLike)
    assert scaled.values == pytest.approx([au.ANGSTROM, 2.0 * au.ANGSTROM])


@pytest.mark.parametrize("bad_value", [True, "1", None, object(), {"x": 1.0}])
def test_rejects_unsupported_annotated_values_with_context(bad_value: Any) -> None:
    with pytest.raises(MappingConversionError) as caught:
        convert_mapping({"dt[fs]": bad_value}, au)
    assert "$.dt" in str(caught.value)
    assert "dt[fs]" in str(caught.value)


@pytest.mark.parametrize("bad_key", ["dt[fs", "dt[]", "[]", "dt[fs]tail"])
def test_rejects_malformed_annotations_with_original_key_and_path(bad_key: str) -> None:
    with pytest.raises(MappingConversionError, match=r"\$.*" + bad_key.replace("[", r"\[")):
        convert_mapping({bad_key: 1.0}, au)


def test_unknown_unit_reports_expression_original_key_and_path() -> None:
    with pytest.raises(MappingConversionError) as caught:
        convert_mapping({"options": {"dt[fortnight]": 1.0}}, au)

    message = str(caught.value)
    assert "$.options" in message
    assert "dt[fortnight]" in message
    assert "fortnight" in message


@pytest.mark.parametrize(
    "source",
    [
        {"dt": 1.0, "dt[fs]": 2.0},
        {"dt[fs]": 1.0, "dt": 2.0},
        {"dt[fs]": 1.0, "dt[ps]": 2.0},
    ],
)
def test_detects_key_collisions_independent_of_input_order(source: Mapping[str, float]) -> None:
    with pytest.raises(MappingConversionError, match="normalized key collision"):
        convert_mapping(source, au)


def test_unannotated_canonical_and_annotated_values_are_not_silently_overwritten() -> None:
    mapping = {"group": {"dt": 1.0, "dt[fs]": 2.0}}

    with pytest.raises(MappingConversionError) as caught:
        convert_mapping(mapping, au)

    assert "$.group.dt" in str(caught.value)


def test_requires_mapping_and_unit_system_arguments() -> None:
    with pytest.raises(TypeError, match="mapping"):
        convert_mapping([], au)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="UnitSystem"):
        convert_mapping({}, object())  # type: ignore[arg-type]


def test_accepts_a_non_atomic_unit_system() -> None:
    md = UnitSystem(length="angstrom", time="ps", energy="kcal/mol")

    assert convert_mapping({"dt[fs]": 500.0}, md)["dt"] == pytest.approx(0.5)


def test_reference_units_validate_dimensions_without_selecting_conversion_target() -> None:
    md = UnitSystem(length="angstrom", time="ps", energy="kcal/mol")

    converted = convert_mapping(
        {"dt[fs]": 0.5, "another_dt[ps]": 1.0, "free[angstrom]": 2.0},
        md,
        reference_units={"/dt": "fs", "/another_dt": "fs"},
    )

    assert converted["dt"] == pytest.approx(0.0005)
    assert converted["another_dt"] == pytest.approx(1.0)
    assert converted["free"] == pytest.approx(2.0)


def test_reference_units_accept_bare_top_level_names() -> None:
    converted = convert_mapping(
        {"dt[fs]": 0.5, "qtb": {"tseg[ps]": 2.0}},
        au,
        reference_units={"dt": "ps", "qtb/tseg": "fs"},
    )

    assert converted["dt"] == pytest.approx(0.5 * au.FS)
    assert converted["qtb"]["tseg"] == pytest.approx(2.0 * au.PS)


def test_reference_mismatch_reports_path_expressions_and_dimensions() -> None:
    md = UnitSystem(length="angstrom", time="ps", energy="kcal/mol")

    with pytest.raises(MappingConversionError) as caught:
        convert_mapping(
            {"simulation": {"dt[eV]": 0.5}},
            md,
            reference_units={"/simulation/dt": "fs"},
        )

    message = str(caught.value)
    assert "/simulation/dt" in message
    assert "eV" in message and "fs" in message
    assert "M^1 L^2 T^-2" in message and "T^1" in message


def test_reference_paths_distinguish_equal_nested_leaf_names() -> None:
    result = convert_mapping(
        {"left": {"value[fs]": 1.0}, "right": {"value[eV]": 1.0}},
        au,
        reference_units={"/left/value": "ps", "/right/value": "Hartree"},
    )

    assert result["left"]["value"] == pytest.approx(au.FS)
    assert result["right"]["value"] == pytest.approx(au.EV)


def test_reference_paths_escape_slashes_and_tildes_in_field_names() -> None:
    result = convert_mapping(
        {"group/name~1": {"dt[fs]": 1.0}},
        au,
        reference_units={"/group~1name~01/dt": "ps"},
    )

    assert result["group/name~1"]["dt"] == pytest.approx(au.FS)


def test_slashless_pointer_uses_json_pointer_escapes_for_slashes_in_field_names() -> None:
    result = convert_mapping(
        {"group/name~1": {"dt[fs]": 1.0}},
        au,
        reference_units={"group~1name~01/dt": "ps"},
    )

    assert result["group/name~1"]["dt"] == pytest.approx(au.FS)


def test_reference_path_does_not_alias_integer_and_string_mapping_keys() -> None:
    result = convert_mapping(
        {1: {"dt[eV]": 1.0}, "1": {"dt[fs]": 1.0}},
        au,
        reference_units={"/1/dt": "fs"},
    )

    assert result[1]["dt"] == pytest.approx(au.EV)
    assert result["1"]["dt"] == pytest.approx(au.FS)


def test_reference_units_accept_arbitrary_compound_dimensions() -> None:
    result = convert_mapping(
        {"momentum[Da*angstrom/ps]": 2.0},
        au,
        reference_units={"/momentum": "eV*ps/angstrom"},
    )

    assert result["momentum"] == pytest.approx(2.0 * au.factor_from("Da*angstrom/ps"))


@pytest.mark.parametrize(
    "references",
    [
        {"/missing": "fortnight"},
        {None: "fs"},
        {"/missing": 1},
        {"/bad~escape": "fs"},
        [],
    ],
)
def test_invalid_reference_schema_fails_even_when_field_is_absent(references: Any) -> None:
    with pytest.raises(ReferenceUnitsError):
        convert_mapping({}, au, reference_units=references)


def test_reference_units_reject_equivalent_bare_and_pointer_paths() -> None:
    with pytest.raises(ReferenceUnitsError, match="duplicate reference path"):
        convert_mapping({}, au, reference_units={"dt": "fs", "/dt": "ps"})


def test_empty_slashless_path_addresses_an_empty_top_level_key() -> None:
    converted = convert_mapping({"[fs]": 1.0}, au, reference_units={"": "ps"})

    assert converted[""] == pytest.approx(au.FS)
