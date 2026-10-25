# Pydantic integration

Install the optional integration extra with `pip install hartreez[pydantic]`.
The core `hartreez` import does not import or require Pydantic. Import
`UnitDimension` from the core package and `UnitAwareModel` from the optional
integration module:

```python
from typing import Annotated

from pydantic import Field

from hartreez import UnitDimension, au
from hartreez.pydantic import UnitAwareModel


class SimulationInput(UnitAwareModel):
    timestep: Annotated[float, UnitDimension("fs")] = Field(gt=0)
    force: Annotated[float, UnitDimension("eV/angstrom")]


config = SimulationInput.model_validate(
    {"timestep[ps]": 0.5, "force[eV/angstrom]": 1.0},
    context={"unit_system": au},
)
```

`config.timestep` and `config.force` contain values converted into `au` before
Pydantic checks their field types, constraints, and application validators.
For example, a positive-value constraint sees the value converted into the
working system.
The model validator removes annotated key suffixes at the current model level
only; nested models validate their own mappings and inherit Pydantic's context.
A `model_validator(mode="before")` that you define on a `UnitAwareModel` runs
before this conversion and receives the raw input, suffixed keys included.
Field validators, including `mode="before"` ones, receive converted values.

### Nested models

Nested models and lists of nested models use the same validation context. Each
model converts only its own annotated fields, so the same leaf name can have a
different dimension in another model:

```python
from typing import Annotated

from pydantic import Field

from hartreez import UnitDimension, UnitSystem
from hartreez.pydantic import UnitAwareModel


class ParticleInput(UnitAwareModel):
    mass: Annotated[float, UnitDimension("Da")] = Field(gt=0)
    momentum: Annotated[float, UnitDimension("Da*angstrom/ps")]


class SimulationInput(UnitAwareModel):
    timestep: Annotated[float, UnitDimension("fs")] = Field(gt=0)
    energy: Annotated[float, UnitDimension("eV")]
    particle: ParticleInput
    particles: list[ParticleInput]


md = UnitSystem(length="angstrom", time="ps", energy="kcal/mol")
config = SimulationInput.model_validate(
    {
        "timestep[fs]": 100.0,
        "energy[eV]": 2.0,
        "particle": {
            "mass[kg]": 1.0,
            "momentum[eV*ps/angstrom]": 3.0,
        },
        "particles": [
            {"mass[Da]": 12.0, "momentum[Da*angstrom/ps]": 4.0},
        ],
    },
    context={"unit_system": md},
)
```

The parent converts its timestep and energy; each `ParticleInput` converts its
mass and momentum. For example, `energy` expects energy while `mass` expects
mass, even though both are processed with the same `UnitSystem`. Conversion
happens before the models' field constraints and validators run. The input
mapping is left unchanged. Already canonical nested input can be validated
without context, while an annotated key at any nesting depth requires the
context. Pydantic reports the enclosing model field and, for list items, the
list index in the error location (for example `particle` or `particles.0`).

## Declare dimensions

`UnitDimension` accepts any supported unit expression, including compounds
such as `Da*angstrom/ps` or `eV/angstrom^3`. It is immutable and records only
the expression's dimensions. Its scale does not choose the output unit. Every
annotated value is converted into the `UnitSystem` supplied in validation
context. As with `reference_units` in [`mappings.md`](mappings.md), two
expressions with equal dimensions are compatible:

```python
from typing import Annotated

from hartreez import UnitDimension
from hartreez.pydantic import UnitAwareModel


class HasTime(UnitAwareModel):
    elapsed: Annotated[float, UnitDimension("fs")]
```

`UnitDimension("fs")` describes time dimensionality; it does not mean that
the field stores femtoseconds. For a canonical payload whose keys have no
unit suffix, validation needs no unit-system context:

```python
already_internal = HasTime.model_validate(
    {"elapsed": 0.0005}
)
```

When input contains any unit-annotated key, provide a usable `UnitSystem` as
`context={"unit_system": system}`. Omitting it or supplying another value
raises a Pydantic `ValidationError` instead of silently leaving values
unconverted. The chosen field dimensions are checked when an annotated value
is supplied. Normalization collisions, such as `timestep` alongside
`timestep[fs]`, and key suffixes incompatible with a field's dimension also
become validation errors.

### Where dimensions can be declared

A unit suffix belongs to a field's input key, so a field has at most one
dimension, and `UnitDimension` can only appear where it describes the whole
field. Name the dimension once as a type alias and use it in these shapes:

```python
from collections.abc import Sequence
from typing import Annotated

from pydantic import Field

from hartreez import UnitDimension
from hartreez.pydantic import UnitAwareModel

Length = Annotated[float, UnitDimension("angstrom")]


class CellInput(UnitAwareModel):
    positions: list[list[Length]]
    cell: tuple[Length, Length, Length]
    shifts: tuple[Length, ...] = ()
    lattice: Sequence[Length] = ()
    cutoff: Length | None = None
    skin: Annotated[Length | None, Field(gt=0)] = None
```

The supported shapes are:

- `Annotated[T, UnitDimension(...)]`, including nested `Annotated`;
- type aliases created with the `type` statement (Python 3.12+) or
  `TypeAliasType`, without type parameters;
- `Optional[T]`, also written `T | None`;
- `list[T]`, `set[T]`, `frozenset[T]`, and `Sequence[T]`;
- `tuple[T, ...]`, and fixed tuples whose elements all have the same
  dimension.

One suffix applies one conversion factor to the whole value, and lists,
tuples, sets, and frozensets are scaled element by element: `positions[nm]`
converts every position. Every `UnitDimension` in a field's annotation must
agree on the dimensions.

Conversion happens before Pydantic validates the field, so a suffixed value
must itself be one of those four containers, nested as needed, or a value
that supports multiplication by a Python `float`, such as a number or a NumPy
array. Other iterables that Pydantic would accept for the field, such as a
`deque` or a `range` for `Sequence[Length]`, raise a `ValidationError` naming
their type; convert them to a list first. Arrays parsed from JSON or TOML
are always lists.

A `UnitDimension` anywhere else raises `TypeError` when the class is defined,
independently of any input. This includes `dict[str, Length]`,
`deque[Length]`, unions other than `Optional`, such as `Length | str` or
`Length | Time`, tuples mixing dimensioned and plain elements, such as
`tuple[Length, int]`, tuples of different dimensions, such as
`tuple[Length, Time]`, and generic type aliases, such as `Vector[Length]`.

Nested `UnitAwareModel` types declare their own fields' dimensions, and can
be nested in any shape, including `dict` values. A nested model must itself
be a `UnitAwareModel` to convert its suffixed keys. A plain `BaseModel`, a
dataclass, or a `TypedDict` would drop them as unknown keys and keep its
defaults. Nesting one that declares `UnitDimension` fields raises `TypeError`
when the outer class is defined, whether it appears directly, in a list,
union, or `dict`, or inside another plain model, dataclass, or `TypedDict`.
A generic one counts when its own fields declare a dimension, as in
`Box[int]`, or when a type argument does, as in `Box[Length]`.

This early check reads the schema Pydantic compiles for the model, so it
cannot see into custom validation code, such as a `PlainValidator`, and it
waits for models with unresolved forward references. Suffixed input those
reach is still rejected when it is validated, as described below.

### Which keys are suffixed

Suffixed keys follow Pydantic's own rules for which input keys fill a field:
a string validation alias, including one set through `alias` or an
`alias_generator`, every string in `AliasChoices`, and the field name when
the field has no alias or the model enables `validate_by_name` (or
`populate_by_name`). A suffixed field name that Pydantic would ignore is
rejected like any other undeclared key.

Fields that share an input key must agree on its dimensions, and either all
declare a `UnitDimension` or none do; otherwise defining the class raises
`TypeError`. The first key of an `AliasPath` counts as an input key of the
field. A field with a `UnitDimension` cannot use `AliasPath`, directly or in
`AliasChoices`, because its suffix would sit below the model's own level;
defining the class raises `TypeError`.

### Suffixes require a declared dimension

A suffixed key is accepted only when it names a field whose annotation
declares a `UnitDimension`. Any other suffixed key at the model's own level,
such as `count[ps]` on an integer field or an unknown key, raises a
`ValidationError` whose message says that no reference unit is declared for
it. A value is never converted without its dimension being checked.
Unsuffixed keys are left to normal Pydantic validation.

Only a `UnitAwareModel` converts its keys, so every suffixed key in the
input must be converted by a `UnitAwareModel` that ends up in the validated
result. Otherwise validation fails with a message naming the key's path.
This covers a key that would stay in a `dict` field, an `Any` field, or an
extra value with its value unconverted, and a key that a plain model,
dataclass, or `TypedDict` would drop as unknown, leaving its default behind,
whatever their `extra` policy. A mapping object used at several places in
the input must be converted at each of them. Mappings are searched for inside
other mappings and any sequence, such as a list, a tuple, or a `deque`; other
iterables, such as generators, are not iterated:

```python
from pydantic import ValidationError

from hartreez import au
from hartreez.pydantic import UnitAwareModel


class Masses(UnitAwareModel):
    masses: dict[str, float] = {}


try:
    Masses.model_validate({"masses": {"H[Da]": 1.008}}, context={"unit_system": au})
except ValidationError as error:
    assert "'H[Da]' at $.masses" in str(error)
```

A nested `UnitAwareModel` is matched to the input mapping it was built
from: the same object, which Pydantic passes on from the input, or else a
copy holding every suffixed entry of that mapping unchanged. A validator that
runs before it, such as a `model_validator(mode="before")` on an enclosing
model or a `field_validator(mode="before")` on the field, may therefore copy
nested input, add or remove unsuffixed keys, or merge in entries from
another part of the input, such as shared defaults:

```python
from typing import Annotated, Any

from pydantic import field_validator

from hartreez import UnitDimension, au
from hartreez.pydantic import UnitAwareModel


class Species(UnitAwareModel):
    name: str
    mass: Annotated[float, UnitDimension("Da")]


class Composition(UnitAwareModel):
    species: list[Species]

    @field_validator("species", mode="before")
    @classmethod
    def from_mapping(cls, value: Any) -> Any:
        # Accept species keyed by name.
        if isinstance(value, dict):
            return [{"name": name, **entry} for name, entry in value.items()]
        return value


water = Composition.model_validate(
    {"species": {"H": {"mass[Da]": 1.008}, "O": {"mass[Da]": 15.999}}},
    context={"unit_system": au},
)
```

A validator must not rename a suffixed key, for example to migrate a legacy
key such as `m[Da]` to `mass[Da]`, or change a suffixed value, even only its
type, such as `True` to `1.0`: the result matches no input mapping, so
validation fails. Do such changes in the nested model's own
`model_validator(mode="before")`, which receives its input after the match.
Suffixed entries that end up in no `UnitAwareModel`, such as defaults that no
nested model receives, are reported like any other unconverted key.

For models that read input files, set `extra="forbid"` at every level as
well, so misspelled keys are rejected like stray suffixed ones.
