# Changelog

All notable changes to `hartreez` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, breaking changes bump
the minor version.

## Unreleased

## 0.2.0 - 2026-10-25

### Removed

- **Breaking:** the Pydantic integration: the `hartreez[pydantic]` extra,
  `hartreez.pydantic.UnitAwareModel` and `hartreez.UnitDimension`
  ([#20](https://github.com/thomasple/hartreez/issues/20)). Convert input with
  `convert_mapping`, then validate the result:

  ```python
  data = convert_mapping(raw, au, reference_units={"energy": "eV"})
  inputs = CalculationInput.model_validate(data)
  ```

  See [Validate with Pydantic or another library](https://github.com/thomasple/hartreez/blob/main/docs/mappings.md#validate-with-pydantic-or-another-library).
