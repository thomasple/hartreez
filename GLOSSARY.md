# hartreez

Unit conversion and coherent working systems for atomistic calculations, where
values stay ordinary numbers and arrays.

## Language

**Working system**:
A coherent set of units, fixed by three mechanical units plus charge and
temperature units, in which a calculation's values are expressed.
_Avoid_: unit set, internal units

**Physical constant**:
A named constant of nature (such as Boltzmann's constant or the reduced Planck
constant) expressed in a working system's units.
_Avoid_: constant (when unit factors are also meant)

**Unit factor**:
One external unit expressed in a working system, so that multiplying a value
in that external unit by it converts the value into the system.
_Avoid_: constant, conversion constant
