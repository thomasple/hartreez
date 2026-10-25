# hartreez

`hartreez` provides unit conversion and coherent working systems for atomistic
calculations. Convert lengths, energies, forces, and other quantities using
familiar units such as angstrom, bohr, eV, and Hartree. Values remain ordinary
numbers and arrays, with no quantity wrappers or core runtime dependencies.

## Install

```sh
pip install hartreez
```

Requires Python 3.10 or newer.

## Work in atomic units

Choose a working system, convert input values into it, and express results in
the units you need. The built-in `au` system uses bohr for length and Hartree
for energy:

```python
from hartreez import au

energy_eV = 1.25
force_eV_per_angstrom = -0.4

energy_au = energy_eV * au.EV
force_au = au.to_internal(force_eV_per_angstrom, "eV/angstrom")

# Express values from the working system in external units.
energy_out_eV = au.from_internal(energy_au, "eV")
force_out_eV_per_angstrom = au.from_internal(force_au, "eV/angstrom")
```

Factors are ordinary floats; a named factor is one external unit expressed
in the active system, so multiply the external value by it. `to_internal`
multiplies by this factor; `from_internal` applies its reciprocal.
Array-like inputs keep control of their arithmetic through scalar
multiplication; the core neither imports nor coerces NumPy, PyTorch, or other
array libraries.

## Choose a working system

Define a coherent system by choosing exactly three mechanical units from
length, time, energy, and mass. `hartreez` derives the fourth using
`E = M L^2 / T^2`. For example, choose angstrom, picosecond, and kcal/mol;
mass is derived:

```python
from hartreez import UnitSystem

units = UnitSystem(length="angstrom", time="ps", energy="kcal/mol")

energy = units.to_internal(2.0, "eV")
time_ps = units.to_internal(0.5, "fs")
energy_out_eV = units.from_internal(energy, "eV")
thermal_energy = 300.0 * units.K_B  # kcal/mol at 300 K
```

`units.K_B` is Boltzmann's constant in this system. Other physical constants,
such as `au.HBAR`, are available on the corresponding system as ordinary
floats; see [physical constants](https://github.com/thomasple/hartreez/blob/main/docs/constants.md).

## Unit conventions

- `mol` represents Avogadro's particle count and is dimensionless, so
  `kcal/mol` and `kJ/mol` are energy units.
- Spectroscopic `cm1` is cyclic frequency (`f = c*wavenumber`). Use `1/cm`
  or `cm^-1` for geometric inverse length; multiply by `2*pi` when angular
  frequency is needed.

See the [unit reference](https://github.com/thomasple/hartreez/blob/main/docs/units.md) for supported names, expression syntax,
and details of these conventions.

## Convert mappings

Convert unit-annotated values in an input mapping into your working system:

```python
from hartreez import UnitSystem, convert_mapping

units = UnitSystem(length="angstrom", time="ps", energy="kcal/mol")
inputs = convert_mapping({"time[fs]": 0.5, "energy[eV]": 2.0}, units)
assert inputs["time"] == 0.0005
```

See [mapping conversion](https://github.com/thomasple/hartreez/blob/main/docs/mappings.md) for nested data and dimensional
checks.

## Validate inputs with Pydantic

`hartreez` does not depend on a validation library. Convert unit-annotated
input with `convert_mapping`, then validate the plain result with Pydantic or
any other tool. See
[validating with Pydantic](https://github.com/thomasple/hartreez/blob/main/docs/mappings.md#validate-with-pydantic-or-another-library)
for an example.

## Use the command line

```sh
hartreez convert 1 eV Ha --verbose
```

See the [CLI guide](https://github.com/thomasple/hartreez/blob/main/docs/cli.md) for unit factors, constants, and vocabulary
queries.
