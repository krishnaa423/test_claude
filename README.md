# DFT Plane Wave Code

A simple implementation of Density Functional Theory (DFT) using plane wave basis set.

## Features

- Plane wave basis set with energy cutoff
- Norm-conserving pseudopotentials (ONCVPSP UPF format)
- K-point sampling with spglib symmetry reduction
- LDA exchange-correlation functional (with optional pylibxc support)
- SCF solver with Pulay mixing
- Total energy calculation

## Installation

```bash
pip install .
```

For pylibxc support (recommended for production calculations):
```bash
pip install .[libxc]
```

## Usage

```python
from dft_pw import Crystal, DFTCalculator

# Create silicon diamond structure
crystal = Crystal.diamond(5.43, 'Si', units='angstrom')

# Create calculator
calc = DFTCalculator(
    crystal=crystal,
    ecut=10.0,  # Hartree
    kgrid=(4, 4, 4),
    xc='LDA',
    pseudopotential_dir='./pseudopotentials',
)

# Run calculation
result = calc.calculate()
result.print_summary()
```

## Running the Example

```bash
cd examples
python silicon.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
dft_pw/
├── __init__.py         # Package initialization
├── crystal.py          # Crystal structure handling
├── basis.py            # Plane wave basis and FFT grid
├── kpoints.py          # K-point generation with spglib
├── pseudopotential.py  # UPF pseudopotential reader
├── xc.py               # Exchange-correlation functionals
├── hamiltonian.py      # Hamiltonian construction
├── scf.py              # SCF solver
├── calculator.py       # High-level calculator interface
└── cli.py              # Command-line interface
```

## Theory

The code implements the Kohn-Sham equations:

```
[-∇²/2 + V_eff(r)] ψ_nk(r) = ε_nk ψ_nk(r)
```

where `V_eff = V_loc + V_nl + V_H + V_xc` includes:
- Local pseudopotential
- Non-local pseudopotential (Kleinman-Bylander form)
- Hartree potential
- Exchange-correlation potential

Wavefunctions are expanded in plane waves:
```
ψ_nk(r) = Σ_G c_nk(G) exp[i(k+G)·r]
```

## Dependencies

- numpy
- scipy
- spglib
- pylibxc (optional, for advanced XC functionals)

## License

MIT
