### This script measures shear power spectra from shear healpix maps
# Noise spectra are also measured to estimate the shape noise

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
pi = np.pi
from datetime import datetime
projpath = '/net/home/fohlen13/yanza21/research/projects/WL/'
out_path = '/net/home/fohlen13/yanza21/research/projects/WL/results/'
datapath = '/net/home/fohlen13/yanza21/DATA/shearmap/'

import healpy as hp
import pymaster as nmt

# load files 

import yaml
from yaml import Loader
config_file = sys.argv[1]

with open(config_file) as file:
    documents = yaml.load(file, Loader=Loader)
    
nside = documents['Nside']

shear_healpix_path = documents['shear_healpix_path']
outlabel = documents['shear_healpix_path']
maskfile = documents['mask_file']

shear_output_filenames = {"triplet" :        os.path.join(shear_healpix_path, "triplet_C.fits"),
                          "singlet_mask" :   os.path.join(shear_healpix_path, "singlet_mask_C.fits"),
                          "doublet_mask" :   os.path.join(shear_healpix_path, "doublet_mask_C.fits"),
                          "doublet_weight" : os.path.join(shear_healpix_path, "doublet_weight_C.fits")}

shear_noise_healpix_path = shear_healpix_path[:-1] + '_noise'
shear_output_noise_filenames = {"triplet" :   os.path.join(shear_noise_healpix_path, "triplet_C.fits"),
                          "singlet_mask" :   os.path.join(shear_noise_healpix_path, "singlet_mask_C.fits"),
                          "doublet_mask" :   os.path.join(shear_noise_healpix_path, "doublet_mask_C.fits"),
                          "doublet_weight" : os.path.join(shear_noise_healpix_path, "doublet_weight_C.fits")}


if maskfile == 'infile':
    lensing_mask = hp.read_map(shear_output_filenames["doublet_weight"])
else:
    doublet = hp.read_map(shear_output_filenames["doublet_weight"])
    extmask = hp.read_map(maskfile)
    if extmask.size != doublet.size:
        extmask = hp.ud_grade(extmask, hp.npix2nside(doublet.size))
    
    lensing_mask = extmask * doublet
    
mask =  nmt.mask_apodization(lensing_mask, 1, apotype="C2")
smap = hp.read_map(shear_output_filenames["triplet"], field=[1, 2])
smap_noise = hp.read_map(shear_output_noise_filenames["triplet"], field=[1, 2])

# Unify Nside for maps and masks

if mask.size != hp.nside2npix(nside):
    print('Mask converting to Nside = '+str(nside))
    mask = hp.ud_grade(mask, nside)
if smap[0].size != hp.nside2npix(nside):
    print('Shear map converting to Nside = '+str(nside))
    smap = hp.ud_grade(smap, nside)
if smap_noise[0].size != hp.nside2npix(nside):
    print('Noise map converting to Nside = '+str(nside))
    smap_noise = hp.ud_grade(smap_noise, nside)    

# NaMaster setup
    
f = nmt.NmtField(mask, np.array([smap[0], smap[1]]))
f_noise =nmt.NmtField(mask, np.array([smap_noise[0], smap_noise[1]]))

#b = nmt.NmtBin.from_nside_linear(Ns, 300)

bin_edge_low = documents['binning']['bin_edge_low']
bin_edge_high = min(3*nside, documents['binning']['bin_edge_high'])
bin_number = documents['binning']['bin_number']

lbinedges = np.int64(np.geomspace(bin_edge_low, bin_edge_high,
                                  bin_number))
b = nmt.NmtBin.from_edges(lbinedges[:-1], lbinedges[1:])

cl_master = nmt.compute_full_master(f, f, b, n_iter=10)
cl_noise_master = nmt.compute_full_master(f_noise, f_noise, b, n_iter=10)

w22 = nmt.NmtWorkspace()
w22.compute_coupling_matrix(f, f, b)

cw = nmt.NmtCovarianceWorkspace()
cw.compute_coupling_coefficients(f, f, f, f)
ell_arr = b.get_effective_ells()

n_ell = ell_arr.size

# Calculate the theoretical Gaussian covariance from continuous power spectra

cl_file = np.loadtxt(out_path + 'cl_shear_polspice_CFIS1500_4096_noise_shearmask.cl')
ell = cl_file.T[0][:3*nside]
ee_cell_noise = np.mean(cl_file.T[2][:3*nside]) 
bb_cell_noise = np.mean(cl_file.T[3][:3*nside])
eb_cell_noise = cl_file.T[6][:3*nside]

cl_file = np.loadtxt(out_path + 'cl_shear_polspice_CFIS1500_4096_noise_shearmask.cl')
ell = cl_file.T[0][:3*nside]
cl_ee = cl_file.T[2][:3*nside] #- ee_cell_noise
cl_bb = cl_file.T[3][:3*nside] #- bb_cell_noise
cl_eb = cl_file.T[6][:3*nside]

covar_22_22 = nmt.gaussian_covariance(cw, 2, 2, 2, 2,  # Spins of the 4 fields
                                      [cl_ee, cl_eb,
                                       cl_eb, cl_bb],  # EE, EB, BE, BB
                                      [cl_ee, cl_eb,
                                       cl_eb, cl_bb],  # EE, EB, BE, BB
                                      [cl_ee, cl_eb,
                                       cl_eb, cl_bb],  # EE, EB, BE, BB
                                      [cl_ee, cl_eb,
                                       cl_eb, cl_bb],  # EE, EB, BE, BB
                                      w22, wb=w22).reshape([n_ell, 4,
                                                            n_ell, 4])
covar_EE_EE = covar_22_22[:, 0, :, 0]
covar_BB_BB = covar_22_22[:, 3, :, 3]

# Save the results

output = documents['output']['outpath'] + 'cl_shear_namaster_' + documents['output']['outlabel']

print('C_ell file saved to: '+output+'.npy')
print('Covmat file saved to: '+output+'_cov.npy')

np.save(output+'.npy', np.vstack([ell_arr, cl_master, cl_noise_master]))
np.save(output+'_cov.npy', covar_22_22)
