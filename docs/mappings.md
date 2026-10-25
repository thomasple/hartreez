# Mapping conversion

Use `convert_mapping` at a configuration or object construction boundary to
convert annotated values into a selected working `UnitSystem`. It accepts
Python mappings directly, so the caller can use the same API after loading
YAML, TOML, JSON, or any other format. `hartreez` does not load or write those
formats itself.

```python
from hartreez import au, convert_mapping

raw = {
    "timestep[fs]": 0.5,
    "cell[angstrom]": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
    "solver": {"tolerance": 1e-8},
}
config = convert_mapping(raw, au)
```

`config` has keys `timestep`, `cell`, and `solver`. Annotated values are
multiplied by their unit factor in `au`; nested mappings are normalized
recursively. The input mapping is left untouched. Unannotated values are
assumed to already use the selected working units, and non-string keys are
preserved as-is.

Lists and tuples under annotated keys are scaled recursively while retaining
their structure. Other numeric or array-like values can participate by
supporting multiplication by a Python `float`. The core package does not
import NumPy or another numerical framework. Annotated booleans, strings,
`None`, unsupported leaves, and mapping values raise `MappingConversionError`
with the affected path and original key.

Annotations use a terminal unit expression, for example `dt[fs]`,
`cell[angstrom]`, or `force[eV/angstrom]`. The annotation is removed after
conversion. If multiple input keys normalize to the same name, conversion
raises `MappingConversionError` instead of overwriting a value.

```python
from hartreez import MappingConversionError, au, convert_mapping

try:
    config = convert_mapping({"dt[fortnight]": 1.0}, au)
except MappingConversionError as error:
    print(error)
```

## Validate expected dimensions

Pass an optional `reference_units` mapping to validate the dimensions of
selected annotated fields. The references do not choose the output units:
every annotated value is still converted into the supplied `UnitSystem`.
For example, a timestep annotated in femtoseconds or picoseconds is accepted
against an `fs` reference, then converted to the system's time unit:

```python
from hartreez import UnitSystem, convert_mapping

md = UnitSystem(length="angstrom", time="ps", energy="kcal/mol")
config = convert_mapping(
    {"dt[fs]": 0.5, "other_dt[ps]": 1.0},
    md,
    reference_units={"dt": "fs", "other_dt": "fs"},
)
assert config["dt"] == 0.0005
assert config["other_dt"] == 1.0
```

Reference keys are JSON Pointers to **normalized** field paths, and the
leading `/` may be omitted. Nested mappings use one token per key;
`~` and `/` within a key are escaped as `~0` and `~1`, respectively. For
example, `{"left/dt": "fs", "right/dt": "ps"}` gives same-named fields
different expectations, and a key literally named `group/name~1` is written
as `group~1name~01`. In a list or tuple of mappings, include its zero-based
index as a token, such as `replicas/0/dt`.

The empty path addresses a top-level key whose name is the empty string; use
`"/"` for the equivalent explicit pointer. An annotated empty-name field is
written as `"[fs]"`. Reference paths address fields, not the mapping document
itself.

Reference paths address string mapping keys and zero-based list or tuple
indices. Python mappings can contain non-string keys, but JSON Pointer does
not encode mapping-key types. To avoid treating an integer key such as `1` as
the string key `"1"`, references are ignored beneath any non-string mapping
key, including when that key appears inside a list or tuple. Mapping conversion
still preserves non-string keys and recursively normalizes their values.

Each value is any supported unit expression, including compound expressions
such as `Da*angstrom/ps`. Only fields with references are checked; other
annotated values are converted normally, while unannotated values remain in
the supplied system's units. Every reference definition is parsed before the
input is traversed, including references to fields absent from the input.
Invalid paths or unit expressions raise `ReferenceUnitsError` (a schema or
configuration error). A supplied annotation whose dimensions disagree with
its reference raises `MappingConversionError` with the normalized path, both
expressions, and both dimensions.

## Validate with Pydantic or another library

`hartreez` has no validation-framework integration. Convert the mapping first,
then pass the result to your validator. Values arrive as plain numbers in the
working system, so field constraints and validators see converted values:

```python
from pydantic import BaseModel, Field

from hartreez import au, convert_mapping


class CalculationInput(BaseModel):
    timestep: float = Field(gt=0)
    energy: float


raw = {"timestep[fs]": 0.5, "energy[eV]": 2.0}
data = convert_mapping(raw, au, reference_units={"timestep": "fs", "energy": "eV"})
inputs = CalculationInput.model_validate(data)
```

`reference_units` declares the expected dimensions. Use JSON Pointer paths for
nested models, for example `{"thermostat/tau": "fs"}`. Only listed paths are
checked: a suffixed key without a reference, such as `junk[ps]`, is converted
unchecked, and Pydantic ignores unknown keys by default. Set
`model_config = ConfigDict(extra="forbid")` on your models to reject them.
Unannotated input is assumed to already be in the working system.
