# Physical constants

`UnitSystem` exposes physical constants as ordinary Python `float` attributes.
Each dimensional value is expressed in that system's coherent units, so it can
be used directly in numerical formulas. `ALPHA` and `N_A` are dimensionless and
retain their stated values in every system; `N_A` is a particle count, matching
the library's dimensionless `mol` convention.

```python
import math

from hartreez import UnitSystem, au

md = UnitSystem(length="angstrom", time="ps", energy="kcal/mol")
thermal_energy = 300.0 * md.K_B
one_ev_in_kcal_per_mol = md.EV  # Unit factors and constants are both floats.

assert math.isclose(au.HBAR, 1.0)
assert math.isclose(au.M_E, 1.0)
assert math.isclose(au.E_CHARGE, 1.0)
```

## Public constants

| Attribute | Definition and dimensions | Provenance |
| --- | --- | --- |
| `K_B` | Boltzmann constant, energy / temperature | Exact SI defining value, `1.380649e-23 J/K` |
| `HBAR` | Reduced Planck constant, energy × time | `PLANCK / (2 pi)` |
| `PLANCK` | Planck constant, energy × time | Exact SI defining value, `6.62607015e-34 J s` |
| `C_LIGHT` | Speed of light, length / time | Exact SI defining value, `299792458 m/s` |
| `ALPHA` | Fine-structure constant, dimensionless | CODATA 2022 central value, `7.2973525643e-3` |
| `M_E` | Electron mass | CODATA 2022, `9.1093837139e-31 kg` |
| `M_P` | Proton mass | CODATA 2022, `1.67262192595e-27 kg` |
| `E_CHARGE` | Elementary charge magnitude | Exact SI defining value, `1.602176634e-19 C` |
| `N_A` | Avogadro particle count, dimensionless | Exact SI defining value, `6.02214076e23` |
| `EPSILON_0` | Vacuum permittivity, charge² / (energy × length) | Derived as `e² / (4 pi alpha hbar c)` |
| `K_E` | Coulomb constant, energy × length / charge² | Derived as `1 / (4 pi epsilon_0)` |

Measured inputs are pinned to the [NIST CODATA 2022
release](https://physics.nist.gov/cuu/Constants/index.html), and exact inputs
follow the [BIPM SI defining constants](https://www.bipm.org/en/measurement-units/si-defining-constants).
Derived constants share these inputs rather than repeating rounded table
values. The source values and immutable dimensioned registry are in
`hartreez.constants`, separately from the unit registry.

The default `au` uses bohr, atomic time, Hartree, elementary charge, and kelvin.
This makes `au.HBAR`, `au.M_E`, `au.E_CHARGE`, `au.BOHR`, and `au.HARTREE`
approximately one. The electrostatic constants satisfy `au.K_E ≈ 1` and
`au.EPSILON_0 ≈ 1/(4 pi)` under the SI coherent Coulomb convention.

For an MD system using `kcal/mol` and kelvin, `K_B` is about
`0.00198720425864083 kcal mol^-1 K^-1`; for example, multiplying by a
temperature in kelvin gives thermal energy in the selected energy unit.

## Tracing and compilation

Physical constants and unit factors (the uppercase attributes and
`constant(name)`) are computed once when a `UnitSystem` is created and stored
as plain floats. Reading them never re-enters the dimensional machinery, so
they are safe inside `torch.compile(..., fullgraph=True)`:

```python
md = UnitSystem(length="angstrom", time="ps", energy="kcal/mol")


@torch.compile(fullgraph=True)
def thermal_energy(temperature):
    return temperature * md.K_B
```

The expression-based methods `factor_from`, `factor_to`, `to_internal`, and
`from_internal` are also known to trace under `torch.compile`, but they parse
the unit expression during tracing, which relies on the tracer supporting
that Python code. For robustness inside compiled regions, prefer the
attributes, or compute expression factors before entering the compiled region.
