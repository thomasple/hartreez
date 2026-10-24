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
`timestep[fs]`, and incompatible annotations also become validation errors.

Optional fields can carry the dimension inside the union, as in
`Annotated[float, UnitDimension("K")] | None` or
`Optional[Annotated[float, UnitDimension("K")]]`; a suffixed value is then
checked like a required field's. If several union members declare
`UnitDimension`, they must agree on the dimensions, otherwise validation of
annotated input raises `TypeError`.

Fields without `UnitDimension` still follow ordinary mapping conversion when
their input key is annotated and a system is supplied. Unannotated fields are
left to normal Pydantic validation. Advanced validation-alias interactions
are not covered by this integration's initial contract.
