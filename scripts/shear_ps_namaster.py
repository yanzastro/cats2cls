# This script measures shear power spectra from shear healpix maps
# Noise spectra are also measured to estimate the shape noise

import numpy as np
import sys
import os
import yaml
from yaml import Loader
import pyccl as ccl
from scipy.special import gamma
import healpy as hp
sys.path.append('/net/home/fohlen13/yanza21/anaconda3/envs/python36/lib/python3.6/site-packages/')
import pymaster as nmt

os.environ['OMP_NUM_THREADS'] = '20'

pi = np.pi
projpath = '/net/home/fohlen13/yanza21/research/projects/WL/'
out_path = '/net/home/fohlen13/yanza21/research/projects/WL/results/'
datapath = '/net/home/fohlen13/yanza21/DATA/shearmap/'

# load files

config_file = sys.argv[1]

with open(config_file) as file:
    documents = yaml.load(file, Loader=Loader)

nside = documents['Nside']

shear_healpix_path = documents['shear_healpix_path']
outlabel = documents['shear_healpix_path']
maskfile = documents['mask_file']

shear_output_filenames = {"triplet": os.path.join(shear_healpix_path,
                                                  "triplet.fits"),
                          "triplet_random": os.path.join(shear_healpix_path,
                                       "triplet_random.fits"),
                          "singlet_mask": os.path.join(shear_healpix_path, "singlet_mask.fits"),
                          "doublet_mask": os.path.join(shear_healpix_path, "doublet_mask.fits"),
                          "doublet_weight": os.path.join(shear_healpix_path,
                                                         "doublet_weight.fits")}


weight = hp.read_map(shear_output_filenames["doublet_weight"])
lensing_mask = hp.read_map(shear_output_filenames["doublet_mask"])

if maskfile == 'infile':
    mask = lensing_mask

else:
    extmask = hp.read_map(maskfile)
    if extmask.size != lensing_mask.size:
        extmask = hp.ud_grade(extmask, hp.npix2nside(lensing_mask.size))

    mask = extmask * lensing_mask

mask_apod = mask  # nmt.mask_apodization(mask, 1, apotype="C2")
whole_mask = weight * mask_apod

#mask = lensing_mask # nmt.mask_apodization(lensing_mask, 1, apotype="C1")
smap = hp.read_map(shear_output_filenames["triplet"], field=(1, 2))
nmap = hp.read_map(shear_output_filenames["triplet_random"], field=(1, 2))
noise_bias = np.load(os.path.join(shear_healpix_path, "noise_bias.npy"))
shape_noise = noise_bias / np.mean(weight**2)

# Unify Nside for maps and masks

if mask.size != hp.nside2npix(nside):
    print('Mask converting to Nside = '+str(nside))
    mask = hp.ud_grade(mask, nside)
if smap[0].size != hp.nside2npix(nside):
    print('Shear map converting to Nside = '+str(nside))
    smap = hp.ud_grade(smap, nside)

# NaMaster binning setup

bin_edge_low = documents['binning']['bin_edge_low']
bin_edge_high = min(3*nside, documents['binning']['bin_edge_high'])
bin_number = documents['binning']['bin_number']

lbinedges = np.int64(np.geomspace(bin_edge_low, bin_edge_high,
                                  bin_number))
b = nmt.NmtBin.from_edges(lbinedges[:-1], lbinedges[1:])

# NaMaster field setup

f = nmt.NmtField(whole_mask, smap)
f_n = nmt.NmtField(whole_mask, nmap)

w = nmt.NmtWorkspace()
w.compute_coupling_matrix(f, f, b)

print('Calculating band powers....')

cl_coupled = nmt.compute_coupled_cell(f, f)
ncl_coupled = nmt.compute_coupled_cell(f_n, f_n)
noise_spec_cov = ncl_coupled / np.mean(weight**2)

# Decouple power spectrum into bandpowers inverting the coupling matrix
cl_master = w.decouple_cell(cl_coupled - ncl_coupled) 

f0 = nmt.NmtField(whole_mask, np.array([whole_mask]))
w00 = nmt.NmtWorkspace()
w00.compute_coupling_matrix(f0, f0, b)
ell_arr = b.get_effective_ells()
n_ell = ell_arr.size

print('Calculating covariance matrices....')

# Create new Cosmology object with a given set of parameters. This keeps track
# of previously-computed cosmological functions

m = 22.893147

def n_z(z, m):
    A = -0.4154*m**2+19.1734*m-220.261
    z0 = 0.1081*m-1.9417
    alpha = 1.79
    sigma=1.3
    nu = 1
    n = A*z**alpha*np.exp(-(z/z0)**alpha)/(z0**(alpha+1)
                                           / alpha*gamma((alpha+1)/alpha)) + \
        (1-A)*np.exp(-(z-nu)**2/2/sigma**2)/2.5046
    return n


z_n = np.linspace(0, 5, 100)
n = n_z(z_n, m)

cosmo = ccl.Cosmology(Omega_c=0.2589, Omega_b=0.025, h=0.6774, sigma8=0.8159,
                      n_s=0.96, T_CMB=2.7255,
                      transfer_function='eisenstein_hu',
                      baryons_power_spectrum='bcm')

ell_cont = np.arange(3*nside)

lens1 = ccl.WeakLensingTracer(cosmo, dndz=(z_n, n))
lens2 = ccl.WeakLensingTracer(cosmo, dndz=(z_n, n))

cls = ccl.angular_cl(cosmo, lens1, lens2, ell_cont)

cl_ee = cl_coupled[0] / np.mean(weight**2)
cl_bb = cl_coupled[3] / np.mean(weight**2)

cw = nmt.NmtCovarianceWorkspace()
cw.compute_coupling_coefficients(f0, f0, f0, f0)

covar_EE_EE = nmt.gaussian_covariance(cw, 0, 0, 0, 0,  # Spins of the 4 fields
                                      [cl_ee],
                                      [cl_ee],
                                      [cl_ee],
                                      [cl_ee],
                                      w00, wb=w00).reshape([n_ell, n_ell])

covar_BB_BB = nmt.gaussian_covariance(cw, 0, 0, 0, 0,  # Spins of the 4 fields
                                      [cl_bb],
                                      [cl_bb],
                                      [cl_bb],
                                      [cl_bb],
                                      w00, wb=w00).reshape([n_ell, n_ell])

# Save the results

output = documents['output']['outpath'] + \
    'cl_shear_namaster_' + documents['output']['outlabel']

print('C_ell file saved to: '+output+'.npy')
print('EE_EE Covmat file saved to: '+output+'_cov_EE_EE.npy')

np.save(output+'.npy', np.vstack([ell_arr, cl_master]))
np.save(output+'_cov_EE_EE.npy', covar_EE_EE)
np.save(output+'_cov_BB_BB.npy', covar_BB_BB)
