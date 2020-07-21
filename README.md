# ocp-modules

This library provides a set of modules to assemble optimal control problems (OCPs) by generating
symbolic expressions based on [CasADi](https://web.casadi.org/).

## Installation

### For Users
ocp-modules is (SOON) available via `pip`:

```
pip3 install --user ocp-modules
```

### For Developers
1. Install `setuptools` and `virtualenv`:
```
pip3 install --user setuptools virtualenv
```

2. Create a new virtual environment in the folder `venv` and source it:
```
git clone <this_repo>
cd <this_repo>
python3 -m virtualenv venc
source ./venv/bin/activate
```

3. Install ocp-modules via symlink into the virtual environment
```
cd <this_repo>
pip3 install -e .
```

### A note regarding CasADi

Casadi can make use of proprietary solvers such as
[HSL MA57](http://www.hsl.rl.ac.uk/catalogue/ma57.html). Follow the instructions
[here](https://github.com/casadi/casadi/wiki/Obtaining-HSL) to install them and modify the solver
options to use the built-in default.

Common issues with HSL solvers:
* Cannot find `libhsl.so`. In this case, manually create a symlink `libcoinhsl.so` ->
  `libhsl.so` in `/usr/lib`
* Failed to load `libmetis.so`: It is hard to get a working instance of metis (the version packaged
  in Ubuntu 18.04 is not compatible). It seems to work without, but spams into stdout.

## Running the examples
Run examples via:

```
./examples/<folder>/<script>.py
```

## Documentation

The documentation is generated via Sphinx and can be found ~~here~~ *(coming soon)*
