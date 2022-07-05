# cats2cls
Measure shear power spectra in harmonic space from shear catalogues

! This code is still under development !

# Requirements
NaMaster, yaml, astropy

# Usage
Firstly, call `make_shearmap.py` in /scripts with

`python make_shearmap.py shearmap_config.yaml`

to generate shearmap in healpix format;

Then, call `shear_ps_namaster.py` in /scripts with

`python shear_ps_namaster.py shearps_config.yaml`
