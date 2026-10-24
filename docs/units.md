# Unit vocabulary and reference constants

`hartreez` uses an explicit, case-sensitive registry; prefixes are never
inferred. `parse_unit()` accepts registered names in products and quotients,
so expressions such as `kcal/mol`, `eV/angstrom`, and `Da*angstrom/ps` need no
extra registrations.

## Unit expressions and direct conversion

Use `parse_unit()` to inspect a unit or `convert()` to convert a scalar or an
array-like value between compatible units:

```python
from hartreez import convert, parse_unit

force = parse_unit("kg*m/s^2")
assert force.dimensions == parse_unit("N").dimensions
energy_kcal_per_mol = convert(1.0, "eV", "kcal/mol")
```

Expressions use registered names, the dimensionless literal `1`, explicit
products (`*`) and quotients (`/`), parentheses, and powers using `^` or `**`.
Powers may be signed integers (`m^-2`), finite decimals (`m^0.5`), or rational
exponents (`m^1/2`, `m^-3/2`). Multiplication is explicit (`kg*m`, not
`kg m`); numeric coefficients other than `1` are not supported. Names are
case-sensitive. Parsed expressions are cached. Unknown names and malformed
expressions raise `UnitSyntaxError` (with `UnknownUnitError` for unknown
names); `convert()` raises `IncompatibleUnitsError` when dimensions differ.

## Registered spellings

The first spelling in each row is canonical. All following spellings are
explicit aliases and have the same scale and dimensions.

| Quantity | Canonical spelling | Explicit aliases | Definition / provenance |
| --- | --- | --- | --- |
| Dimensionless | `1` | — | SI scale 1 |
| Mass | `kg`, `g` | — | SI kilogram; gram is exactly `1e-3 kg` |
| Atomic mass | `Da` | `atomic_mass_unit`, `amu`, `u` | CODATA 2022 atomic mass constant `m_u`; not proton mass |
| Length | `m`, `cm`, `mm`, `km`, `nm` | `meter`, `centimeter`, `kilometer`, `nanometer` | Exact SI decimal scales |
| Angstrom | `angstrom` | `ang`, `Å`, `Angstrom` | Exactly `1e-10 m` |
| Bohr radius | `bohr` | `a0`, `a_0` | CODATA 2022 `a_0`, derived consistently from shared inputs |
| Time | `s`, `ms`, `us`, `ns`, `ps`, `fs` | `second`, `nanosecond`, `picosecond`, `femtosecond` | Exact SI decimal scales |
| Atomic time | `atomic_time` | `atomic_unit_of_time`, `aut` | `hbar / E_h`, using CODATA 2022 derived atomic units |
| Charge | `C` | `coulomb` | SI coulomb |
| Elementary charge | `e` | `elementary_charge` | Exact SI defining value `1.602176634e-19 C` |
| Temperature | `K` | `kelvin` | SI kelvin |
| Energy | `J` | `joule` | SI joule |
| Electronvolt | `eV` | `electronvolt`, `ev` | Exact `e joule` |
| Millielectronvolt | `meV` | `mev` | Exactly `1e-3 eV` |
| Hartree | `Hartree` | `Hartree_energy`, `Ha` | CODATA 2022 `E_h`, derived as `alpha^2 m_e c^2` |
| Rydberg | `Rydberg` | `Ry` | Exactly one half Hartree |
| Thermochemical calorie | `cal` | `cal_th` | Exactly `4.184 J` |
| Thermochemical kilocalorie | `kcal` | `kcal_th`, `kilocalorie` | Exactly `4184 J` |
| Kilojoule | `kJ` | `kilojoule` | Exactly `1000 J` |
| Frequency | `Hz` | `hertz` | `s^-1` |
| Terahertz | `THz` | `terahertz` | Exactly `1e12 Hz` |
| Spectroscopic inverse centimeter | `cm1` | — | Scale `c * 100 Hz`; see below |
| Pressure | `Pa`, `bar`, `kbar`, `atm`, `GPa` | `pascal`, `atmosphere` | `1 bar = 1e5 Pa`, `1 kbar = 1e8 Pa`, `1 atm = 101325 Pa`, `1 GPa = 1e9 Pa` |
| Force | `N`, `nN` | `newton` | SI newton; nanonewton is exactly `1e-9 N` |
| Dipole | `D` | `debye` | `1e-21/c C m` under the Debye definition |
| Mole count | `mol` | — | Dimensionless scale `N_A = 6.02214076e23` |
| Time convenience | `min`, `h` | — | Exactly 60 and 3600 seconds |
| Molar energy expressions | — | `kcalpermol` → `kcal/mol`, `kJpermol` → `kJ/mol` | Aliases for ordinary parsed expressions; neither expression is a canonical unit |

An unlisted spelling such as `kPa`, `angstroms`, or `EV` raises
`UnknownUnitError`. Names preserve case. No runtime registration is available.

## Atomistic mole convention

`mol` represents Avogadro's particle-count multiplicity. It is dimensionless,
with scale `N_A`, so `kcal/mol` and `kJ/mol` retain energy dimensions. This
makes molar energy conversion ordinary dimensional conversion:

```python
from hartreez import convert

convert(1.0, "eV", "kcal/mol")       # about 23.0605478306
convert(1.0, "Hartree", "kJ/mol")   # about 2625.49963945
```

The public Avogadro count is dimensionless; `mol` does not introduce a new
amount-of-substance dimension.

## Coherent working systems

`UnitSystem` requires exactly three of `length`, `time`, `energy`, and `mass`.
The fourth scale is derived from `E = M L^2 / T^2`. Each supplied expression
must have the matching physical dimension. Charge defaults to `e` and
temperature defaults to `K`; either can be overridden with a compatible unit.

```python
from hartreez import UnitSystem, au

md = UnitSystem(length="angstrom", time="ps", energy="kcal/mol")
force_factor = md.factor_from("eV/angstrom")
momentum_factor = md.factor_from("Da*angstrom/ps")
internal_dt = md.to_internal(0.5, "fs")
external_dt = md.from_internal(internal_dt, "fs")

energy_ha = 3.0 * au.EV
distance_bohr = 1.25 * au.ANG
pressure_au = 10.0 * au.KBAR
force_factor = au.factor_from("eV/ang")
```

The convention is consistent for all attributes: a named unit factor is one
unit expressed in the active system. `factor_from` and every convenience
attribute return Python floats. The convenience attributes are `EV`,
`HARTREE`, `HA`, `RY`, `ANGSTROM`, `ANG`, `BOHR`, `NM`, `FS`, `PS`,
`NS`, `DA`, `AMU`, `KCALPERMOL`, `KJPERMOL`, `DEBYE`, `THZ`, `CM1`, `ATM`,
`BAR`, `KBAR`, `GPA`, `PA`, `NEWTON`, and `NN`. Every attribute equals
`factor_from` for the corresponding registered unit expression.

Unit factor attributes are computed once when the system is created and
stored as plain floats; see [tracing and compilation](constants.md#tracing-and-compilation).

The predefined `au` uses bohr, atomic time, Hartree, elementary charge, and
kelvin; electron mass is derived as the coherent mass scale. Thus
`au.EV` converts eV values to Hartree by multiplication, and `au.ANGSTROM`
converts Å values to bohr by multiplication.

## Spectroscopic inverse centimeters

The exact registered spelling `cm1` is spectroscopic wavenumber represented
as cyclic frequency, with scale `c * 100 Hz`. Thus one `cm1` converts to
`0.0299792458 THz`; multiply by `2*pi` separately when angular frequency is
needed. The spelling `cm^-1` follows ordinary unit algebra and has geometric
inverse-length dimensions, as do `1/cm`, `cm**-1`, and `(cm)^-1`. Powers such
as `cm^-1/2` and `cm^-10` are geometric fractional and integer powers.

## Constant provenance

Measured source values use **CODATA 2022**: the NIST [CODATA constants
index](https://physics.nist.gov/cuu/Constants/index.html) identifies that
release, and its [complete 2022 ASCII table](https://physics.nist.gov/cuu/Constants/Table/allascii.txt)
provides the values for the fine-structure constant, electron/proton masses,
and atomic mass constant. Exact `c`, `h`, `e`, `k_B`, and `N_A` follow the
[BIPM SI defining constants](https://www.bipm.org/en/measurement-units/si-defining-constants).
The thermochemical calorie conversion is specified by [NIST SP 811,
Appendix B.8](https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors/nist-guide-si-appendix-b8).
The Debye definition is given by the [IUPAC Gold Book](https://goldbook.iupac.org/terms/view/D01533).

The source values live independently in `hartreez.constants`; the unit
registry uses them to construct immutable `Unit` values. Derived source
values share those inputs: `hbar = h/(2 pi)`, `a_0 = hbar/(m_e c alpha)`,
`E_h = alpha^2 m_e c^2`, atomic time `= hbar/E_h`, and one Rydberg
`= E_h/2`. Dalton uses `m_u` (the atomic mass constant), not `m_p`.
