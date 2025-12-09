# syk-simulation
A library for quantum simulation for the Sachdev–Ye–Kitaev (SYK) model. In addition to the SYK model, we implement a Trotterization and qDrift protocol.

This library is an ongoing research project done as a part of 
[Quantum Open Source Foundation](https://qosf.org/)'s mentorhsip program, cohort 11.

The authors are [Brian Goldsmith](https://github.com/bdg221), 
[Larissa Kroell](https://github.com/lkroell), and [Nishna Aerabati](https://github.com/naerabati).
The mentors are Mariia Mykhailova and Sean Greenway.

# Development

### Setup

Clone repository into `/home/coder/projects` in PsiQDE instance. Then:

```
cd /home/coder/projects/syk-simulation
python -m pip install -e .[dev]  # Install dependencies.
pre-commit install               # Install pre-commit hooks.
pytest                           # Run tests.
```

### Testing

We use pytest for testing. To run all tests, run `pytest`. You can also
run tests in a single file, e.g. `pytest ./syk-simulation/ppr/test_ppr.py`.

### Formatting

This repository uses [Black formatter](https://github.com/psf/black). 
Recommended setup for VSCode is:
* Install extension "[Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)".
* Use Ctrl+Shift+I to format file.

### Pre-commit hooks

This repository uses pre-commit hooks to run formatter and all tests. 
* If "black" hook fails, you need to re-format-code, just run `black .`.
* If "pytest" hook fails, you need to fix failing tests.
* After fixing the issue, commit again.


This is necessary because we currently cannot run tests on Gihub Actions.

In the future, if tests will take too long, we will allow marking some of them 
as "slow", and run only "fast" tests on pre-commit hook.
