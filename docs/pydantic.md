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
For example, a positive-value constraint sees the converted internal value.
The model validator removes annotated key suffixes at the current model level
only; nested models validate their own mappings and inherit Pydantic's context.

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

`UnitDimension` can sit anywhere in a field's annotation, so a dimension
named once as a type alias composes with optional, sequence, and nested
types:

```python
from typing import Annotated

from pydantic import Field

from hartreez import UnitDimension
from hartreez.pydantic import UnitAwareModel

Length = Annotated[float, UnitDimension("angstrom")]


class CellInput(UnitAwareModel):
    positions: list[Length]
    cell: tuple[Length, Length, Length]
    cutoff: Length | None = None
    skin: Annotated[Length | None, Field(gt=0)] = None
```

One suffix applies one conversion factor to the whole value, and lists and
tuples are scaled element by element. A dimension declared on an element type
is therefore the dimension of the whole field: `positions[nm]` converts every
position. Every `UnitDimension` found in a field's annotation must agree on
the dimensions. A field such as `tuple[Length, Time]` cannot be written with
one suffix, so a conflict is a model definition error: defining the class
raises `TypeError`, independently of any input. Aliases created with the
`type` statement (Python 3.12+) or `TypeAliasType` work like plain
assignments. Nested `UnitAwareModel` types declare their own fields'
dimensions, and `dict` values are not searched.

A nested model must itself be a `UnitAwareModel` to convert its suffixed
keys. A plain `BaseModel` would ignore them and keep its defaults, so nesting
one that declares `UnitDimension` fields, directly or in a list, union, or
`dict`, raises `TypeError` when the outer class is defined.

Suffixed keys follow Pydantic's own rules for which input keys fill a field:
a string validation alias, including one set through `alias` or an
`alias_generator`, and the field name when the field has no alias or the model
enables `validate_by_name` (or `populate_by_name`). A suffixed field name that
Pydantic would ignore is rejected like any other undeclared key. Fields that
share an input key must agree on its dimensions, otherwise defining the class
raises `TypeError`. `AliasPath` and `AliasChoices` are not supported.

### Suffixes require a declared dimension

A suffixed key is accepted only when it names a field whose annotation
declares a `UnitDimension`. Any other suffixed key, such as `count[ps]` on an
integer field, an unknown key, or a key reaching a field through
`AliasChoices`, raises a `ValidationError` whose message says that no
reference unit is declared for it. A value is never converted without its
dimension being checked. Unsuffixed keys are left to normal Pydantic
validation.
