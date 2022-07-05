import numpy as np
import matplotlib.pyplot as plt
import sys
import os
pi = np.pi

projpath = '/net/home/fohlen13/yanza21/research/projects/WL/'
datapath = '/net/home/fohlen13/yanza21/DATA/shearmap/'
yamlpath = projpath + 'config_yaml/'

sys.path.append('../source')
import cats2maps
import yaml
from yaml import Loader


config = sys.argv[1]

with open(config) as file:
    documents = yaml.load(file, Loader=Loader)

label = documents['label']
shear_catalogs = documents['shear_catalogs']
shear_columns = documents['shear_columns']
selection = documents['selection']
shear_healpix_path = datapath + label
nside = documents['Nside']

print("Making shear map for "+label+" with Nside="+str(nside))


if ~os.path.isdir(shear_healpix_path):
    os.system('mkdir -p '+shear_healpix_path)
else: pass

z_min = -10000
z_max = 10000

print(selection)

pylenspice.create_shear_healpix_triplet(shear_catalogs=shear_catalogs, 
                                        shear_healpix_path=shear_healpix_path, 
                                        nside=nside, flip_e1=True, 
                                        partial_maps=False, 
                                        c_correction='data', 
                                        m_correction='data', 
                                        selections= selection,
                                        shear_randoms = True,
                                        column_names=shear_columns, 
                                        z_min=z_min, 
                                        z_max=z_max,
                                        hdu_idx=1)
