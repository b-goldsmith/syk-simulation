# syk-simulation
A library for quantum simulation of the Sachdev–Ye–Kitaev (SYK)\[[1](#1-sachdev-ye-kitaev)\] model. This library includes three different techniques for simulating the SYK model: Trotterization\[[2](#2-trotterization)\], qDRIFT\[[3](#3-qdrift)\], and asymmetric qubitization\[[4](#4-asymmetric-qubitization)\] with QSP. 

The library was written using PsiQuantum's [Construct](https://www.psiquantum.com/construct). This includes the PsiQuantum Development Kit [PsiQDK](), Circuit Designer, and Resource Analyzer. In order to use the library directly or in a notebook, either a PsiQuantum QDE must be used or the `psiqworkbench` Python module must be installed. Please see the link above if access is needed.

This library was a research project done as a part of 
[Quantum Open Source Foundation](https://qosf.org/)'s mentorhsip program, cohort 11.

The authors are [Brian Goldsmith](https://github.com/bdg221), 
[Larissa Kroell](https://github.com/lkroell), and [Nishna Aerabati](https://github.com/naerabati).
The mentors are Mariia Mykhailova and Sean Greenway.

## Requirements
### Dependencies

The following dependencies are required to run the library and are included in the pyproject.toml:
python >= 3.12, psiqdk, numpy, scipy, pyqsp

Development dependencies include:
pytest, black>=23.7.0, pre-commit>=3.0.0

The repository includes a `uv.lock` file.

## Setup
### Using uv for a virtual environment

1. Ensure you have Python and `uv` installed
2. Clone the `syk-simultion` repository. (Click [here](https://git-scm.com/) to learn more about `git`)
3. Inside the new `syk-simulation` directory, use `uv` to create a virtual environment (/.venv) and install the required dependencies

```
pip install uv
git clone https://github.com/b-goldsmith/syk-simulation.git
cd syk-simulation
uv sync
```

### Using pip to install dependencies

1. Ensure you have an environment with at least Python 3.12
2. Clone the `syk-simulation` repository. (Click [here](https://git-scm.com/) to learn more about `git`)
3. Inside the new `syk-simulation` directory, run `pip install` to install the local package and the dependencies.  

```
git clone https://github.com/b-goldsmith/syk-simulation.git
cd syk-simulation
python -m pip install -e .'[dev]'
```


## Testing

We use pytest for testing. To run all tests, run `pytest` from the top directory of the project. You can also run tests in a single file, e.g. `pytest ./syk_simulation/ppr/test_ppr.py`.

## Formatting

This repository uses [Black formatter](https://github.com/psf/black). 
Recommended setup for VSCode is:
* Install extension "[Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)".
* Use Ctrl+Shift+I to format file.

## Pre-commit hooks

This repository uses pre-commit hooks to run formatter and all tests. 
* If "black" hook fails, you need to re-format-code, just run `black .`.
* If "pytest" hook fails, you need to fix failing tests.
* After fixing the issue, commit again.

## References:
### [1] Sachdev-Ye-Kitaev
A. Kitaev, “A simple model of quantum holography,”
https://online.kitp.ucsb.edu/online/entangled15/kitaev/, Apr,
May 2015.

### [2] Trotterization 
L. Garc´ıa- ´Alvarez, I. L. Egusquiza, L. Lamata, A. del Campo,
J. Sonner, and E. Solano, “Digital quantum simulation of minimal
AdS/CFT,” Phys. Rev. Lett., vol. 119, p. 040501, Jul 2017. [Online].
Available: https://link.aps.org/doi/10.1103/PhysRevLett.119.040501

### [3] qDRIFT
E. Campbell, “Random compiler for fast hamiltonian simulation,” Phys. 
Rev. Lett., vol. 123, p. 070503, Aug 2019. [Online]. Available:
https://link.aps.org/doi/10.1103/PhysRevLett.123.070503

### [4] Asymmetric Qubitization
R. Babbush, D. W. Berry, and H. Neven, “Quantum simulation
of the sachdev-ye-kitaev model by asymmetric qubitization,”
Phys. Rev. A, vol. 99, p. 040301, Apr 2019. [Online]. Available:
https://link.aps.org/doi/10.1103/PhysRevA.99.040301