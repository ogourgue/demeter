# Demeter

Demeter is a Python package for multiscale biogeomorphic modeling.

## Python environment

Standard packages that must be installed:
* matplotlib
* mpi4py
* numpy
* scipy

### Optional: Telemac support

To use Demeter with Telemac, you need to manually install pputils:
```bash
git clone https://codeberg.org/pprodano/pputils.git
export PYTHONPATH="${PYTHONPATH}:/path/to/pputils"
```
**Note:** Telemac support via pputils may be replaced in future versions.
