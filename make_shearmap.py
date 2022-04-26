import numpy as np
import matplotlib.pyplot as plt
import sys
import os
pi = np.pi

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
    
shear_output_filenames = {"triplet" :        os.path.join(shear_healpix_path, "triplet_C.fits"),
                          "singlet_mask" :   os.path.join(shear_healpix_path, "singlet_mask_C.fits"),
                          "doublet_mask" :   os.path.join(shear_healpix_path, "doublet_mask_C.fits"),
                          "doublet_weight" : os.path.join(shear_healpix_path, "doublet_weight_C.fits")}

z_min = -10000
z_max = 10000

print(selection)

cats2maps.create_shear_healpix_triplet(shear_catalogs=shear_catalogs, 
                                        out_filenames=shear_output_filenames, 
                                        nside=nside, flip_e1=True, 
                                        #convert_to_galactic=True,
                                        partial_maps=False, 
                                        c_correction='data', 
                                        m_correction='data', 
                                        selections= selection,
                                        column_names=shear_columns, 
                                        z_min=z_min, 
                                        z_max=z_max,
                                        hdu_idx=1)

print("Making noise map...")

shear_healpix_path = datapath + label + '_noise'

if ~os.path.isdir(shear_healpix_path):
    os.system('mkdir -p '+shear_healpix_path)
else: pass

shear_output_filenames = {"triplet" :        os.path.join(shear_healpix_path, "triplet_C.fits"),
                          "singlet_mask" :   os.path.join(shear_healpix_path, "singlet_mask_C.fits"),
                          "doublet_mask" :   os.path.join(shear_healpix_path, "doublet_mask_C.fits"),
                          "doublet_weight" : os.path.join(shear_healpix_path, "doublet_weight_C.fits")}


cats2maps.create_shear_healpix_triplet(shear_catalogs=shear_catalogs, 
                                        out_filenames=shear_output_filenames, 
                                        nside=nside, flip_e1=True, 
                                        #convert_to_galactic=True,
                                        partial_maps=False, 
                                        c_correction='data', 
                                        m_correction='data', 
                                        selections= selection,
                                        column_names=shear_columns, 
                                        shear_randoms=True, 
                                        z_min=z_min, 
                                        z_max=z_max,
                                        hdu_idx=1)
