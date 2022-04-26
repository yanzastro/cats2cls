# Functions needed to generate healpix format of galaxy shape maps.

from __future__ import division
import numpy as np

import astropy.io.fits as fits
import astropy.coordinates
import astropy.units as units

import healpy as hp

import subprocess
import sys
import os

pi = np.pi


def write_partial_polarization_file(arrays, nside, filename, mask_value, 
                                    col_names=["kappa", "gamma1", "gamma2"], 
                                    coord="C"):
    def add_partial_header(hdu, nside):
        hdu.header["PIXTYPE"] = "HEALPIX"
        hdu.header["ORDERING"] = "RING"
        hdu.header["COORDSYS"] = coord
        hdu.header["NSIDE"] = nside
        hdu.header["INDXSCHM"] = "EXPLICIT"
        hdu.header["OBJECT"] = "PARTIAL"
        
    primaryhdu = fits.PrimaryHDU()
    hdu_list = [primaryhdu,]
    
    if len(arrays) != 3:
        raise RuntimeError("Number of arrays isn't 3!")
        
    for i in range(3):
        mask = arrays[i] != mask_value
        if np.count_nonzero(mask) == 0:
            mask[0] = 1
        pixel = np.arange(hp.nside2npix(nside))[mask]
        value = arrays[i][mask]
        col_hdu = fits.BinTableHDU.from_columns([fits.Column(name="PIXEL", format="J", array=pixel), 
                                                 fits.Column(name=col_names[i], format="E", array=value)])
        add_partial_header(col_hdu, nside)
        hdu_list.append(col_hdu)

    new_hdu = fits.HDUList(hdu_list)
    new_hdu.writeto(filename, overwrite=True)


def cel2gal_t(ra, dec):
    """Transform equatorial coordinates to galactic coordinates 
    including local rotation angle.
    """

    coords = astropy.coordinates.SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
    coords = coords.transform_to(astropy.coordinates.Galactic)

    pole_coords = astropy.coordinates.SkyCoord(ra=0, dec=90, unit="deg", frame="icrs")
    new_pole_coords = pole_coords.transform_to(astropy.coordinates.Galactic)

    phi = coords.position_angle(new_pole_coords).rad

    return coords.l.deg, coords.b.deg, phi


def cel2gal(ra, dec):
    r = hp.Rotator(coord=['C', 'G'])
    lon, lat = r(ra, dec, lonlat=True)
    new_pole_coords = r(0, 90, lonlat=True)
    new_pole_coords = astropy.coordinates.SkyCoord(ra=new_pole_coords[0],
                                                   dec=new_pole_coords[1],
                                                   unit="deg", frame="icrs")
    coords = astropy.coordinates.SkyCoord(ra=lon, dec=lat, unit="deg", frame="icrs")
    phi = coords.position_angle(new_pole_coords).rad
    return lon, lat, phi

def convert_shear_to_galactic_coordinates(ra, dec, e1, e2):
    lon, lat, delta_phi = cel2gal(ra, dec)
    phi = np.arctan2(e2, e1)/2
    e = np.sqrt(e1**2 + e2**2)
    e1_gal = e*np.cos(2*(phi-delta_phi))
    e2_gal = e*np.sin(2*(phi-delta_phi))

    return lon, lat, e1_gal, e2_gal

def prepare_catalog(catalog_filename, column_names={"ra":"ra", "dec":"dec", 
                                                    "e1":"e1", "e2":"e2", 
                                                    "w":"w", "m":"m", 
                                                    "c1":"c1", "c2":"c2"},
                                 c_correction="data", m_correction="catalog", 
                                 z_min=None, z_max=None, 
                                 selections=[("weight", "gt", 0.0)], hdu_idx=1, has_shear=True):
    """Create healpix shear maps from lensing shape catalogs.

    Required arguments:
    catalog_filename    File name of shape catalog. 
    column_names        Dictionary of column names. Required entries are "ra", 
                        "dec", "e1", "e2", "w". If c_correction="catalog", "c1", 
                        "c2" are required. If m_correction="catalog", "m" is
                        required. Default is {"ra":"ra", "dec":"dec", "e1":"e1", 
                        "e2":"e2", "w":"w", "m":"m", "c1":"c1", "c2":"c2"}.
    c_correction        Apply c correction. Options are "catalog" and "data"
                        (default: "data"). 
    m_correction        Apply m correction. Options are "catalog" or None
                        (default: "catalog").
    z_min               Lower bound for redshift cut (default: None).
    z_max               Upper bound for redshift cut (default: None).
    selections          List of catalog selection criteria. The list entries 
                        consist of tuples with three elements: column name, 
                        operator, and value. The supported operators are "eq", 
                        "neq", "gt", "ge", "lt", "le". Default is 
                        [("weight", "gt", 0.0)].
    hdu_idx             Index of the FITS file HDU to use (default: 1).
    verbose             Verbose output (default: True).
    """
    hdu = fits.open(catalog_filename)

    mask = np.ones(hdu[hdu_idx].data.size, dtype=bool)
    # Apply selections to catalog
    for col, op, val in selections:
        if op == "eq":
            mask = np.logical_and(mask, hdu[hdu_idx].data[col] == val)
        elif op == "neq":
            mask = np.logical_and(mask, hdu[hdu_idx].data[col] != val)
        elif op == "gt":
            mask = np.logical_and(mask, hdu[hdu_idx].data[col] > val)
        elif op == "ge":
            mask = np.logical_and(mask, hdu[hdu_idx].data[col] >= val)
        elif op == "lt":
            mask = np.logical_and(mask, hdu[hdu_idx].data[col] < val)
        elif op == "le":
            mask = np.logical_and(mask, hdu[hdu_idx].data[col] <= val)
        else:
            raise RunTimeError("Operator {} not supported.".format(op))

    # For convenience, redshfit cuts can be applied through the arguments z_min and z_max as well.
    if z_min != None and z_max != None:
        z = hdu[hdu_idx].data[column_names["z"]]
        mask = np.logical_and(mask, np.logical_and(z >= z_min, z < z_max))

    ra = hdu[hdu_idx].data[column_names["ra"]][mask]
    dec = hdu[hdu_idx].data[column_names["dec"]][mask]
    
    if has_shear:
        w = hdu[hdu_idx].data[column_names["w"]][mask]
        e1 = hdu[hdu_idx].data[column_names["e1"]][mask]
        e2 = hdu[hdu_idx].data[column_names["e2"]][mask]

        # Apply c correction
        if c_correction == "catalog":
            # Use c correction supplied by the catalog
            c1 = hdu[hdu_idx].data[column_names["c1"]][mask]
            c2 = hdu[hdu_idx].data[column_names["c2"]][mask]
            c1_mask = c1 > -99
            c2_mask = c2 > -99
            e1[c1_mask] -= c1[c1_mask]
            e2[c2_mask] -= c2[c2_mask]
        elif c_correction == "data":
            # Calculate c correction from the weighted ellipticity average
            c1 = np.sum(w*e1)/np.sum(w)
            c2 = np.sum(w*e2)/np.sum(w)
            e1 -= c1
            e2 -= c2

        # Apply m correction
        if m_correction == "catalog":
            m = hdu[hdu_idx].data[column_names["m"]][mask]
        else:
            m = np.zeros_like(w)
    else:
        e1 = np.ones_like(ra)
        e1 = np.ones_like(ra)
        w = np.ones_like(ra)
        m = np.zeros_like(ra)
    hdu.close()

    return ra, dec, e1, e2, w, m


def calculate_shape_noise(shear_catalogs, nside, 
                            hdu_idx=1, column_names={"ra":"ra", "dec":"dec", 
                                                    "e1":"e1", "e2":"e2", 
                                                    "w":"w", "m":"m", 
                                                    "c1":"c1", "c2":"c2"},
                                 c_correction="data", m_correction="catalog", 
                                 z_min=None, z_max=None, 
                                 selections=[("weight", "gt", 0.0)],
                                 verbose=True, masked_value=0, has_shear=True):
    
    w_map = np.zeros(hp.nside2npix(nside), dtype=np.float32)
    sw2_map = np.zeros(hp.nside2npix(nside), dtype=np.float32)

    for catalog in shear_catalogs:
        if verbose: print("Reading {}.".format(catalog))
        ra, dec, e1, e2, w, m = prepare_shear_catalog(catalog, column_names, 
                                                c_correction, m_correction, 
                                                z_min, z_max,selections, hdu_idx)
        if c_correction == "catalog":
            # Use c correction supplied by the catalog
            c1 = hdu[hdu_idx].data[column_names["c1"]][mask]
            c2 = hdu[hdu_idx].data[column_names["c2"]][mask]
            c1_mask = c1 > -99
            c2_mask = c2 > -99
            e1[c1_mask] -= c1[c1_mask]
            e2[c2_mask] -= c2[c2_mask]
        elif c_correction == "data":
            # Calculate c correction from the weighted ellipticity average
            c1 = np.sum(w*e1)/np.sum(w)
            c2 = np.sum(w*e2)/np.sum(w)
            e1 -= c1
            e2 -= c2
            # Depending on the definition of the coordinate system and ellipticities, one 
            # ellipticity component might have to be flipped.

        pixel_idx = hp.ang2pix(nside, ra, dec, lonlat=True)
        #np.add.at(w_map, pixel_idx, w)
        np.add.at(sw2_map, pixel_idx, w**2*0.5*(e1**2 + e2**2))
    #w_norm = np.sum(w_map)
    #sw2_map[sw2_map!=0] /= w_norm
    Omega = hp.nside2pixarea(nside)
    #fsky = (w_map>0).sum() / (w_map.size*1.0)
    return np.mean(sw2_map) * Omega #/ fsky


def create_shear_healpix_triplet(shear_catalogs, out_filenames, nside, 
                                 flip_e1=False, convert_to_galactic=False, 
                                 partial_maps=True, shear_randoms=False, 
                                 hdu_idx=1, column_names={"ra":"ra", "dec":"dec", 
                                                          "e1":"e1", "e2":"e2", 
                                                          "w":"w", "m":"m", 
                                                          "c1":"c1", "c2":"c2",
                                                          "z":"z"},
                                 c_correction="data", m_correction="catalog", 
                                 z_min=None, z_max=None, has_shear=True,
                                 selections=[("weight", "gt", 0.0)],
                                 verbose=True, masked_value=0):
    """Create healpix shear maps from lensing shape catalogs.

    Required arguments:
    shear_catalogs      List of file names of shape catalogs.
    out_filenames       Dictinary of file names of the output healpix maps. 
                        Required entries are "triplet", "singlet_mask", 
                        "doublet_mask", and "doublet_weight".
    nside               HEALPix nside parameter.

    Optional arguments: 
    flip_e1             Flip e1 component (default: False).
    convert_to_galactic Produce healpix map in Galactic coordinates.
    partial_maps        Write partial maps using explicit indexing 
                        (default: True).
    shear_randoms       Randomize ellipticities (default: False). 
    hdu_idx             Index of the FITS file HDU to use (default: 1).
    column_names        Dictionary of column names. Required entries are "ra", 
                        "dec", "e1", "e2", "w". If c_correction="catalog", "c1", 
                        "c2" are required. If m_correction="catalog", "m" is
                        required. Default is {"ra":"ra", "dec":"dec", "e1":"e1", 
                        "e2":"e2", "w":"w", "m":"m", "c1":"c1", "c2":"c2"}.
    c_correction        Apply c correction. Options are "catalog" and "data"
                        (default: "data"). 
    m_correction        Apply m correction. Options are "catalog" or None
                        (default: "catalog").
    z_min               Lower bound for redshift cut (default: None).
    z_max               Upper bound for redshift cut (default: None).
    selections          List of catalog selection criteria. The list entries 
                        consist of tuples with three elements: column name, 
                        operator, and value. The supported operators are "eq", 
                        "neq", "gt", "ge", "lt", "le". Default is 
                        [("weight", "gt", 0.0)].
    verbose             Verbose output (default: True).
    """

    e1_map = np.zeros(hp.nside2npix(nside), dtype=np.float32)
    e2_map = np.zeros(hp.nside2npix(nside), dtype=np.float32)
    w_map = np.zeros(hp.nside2npix(nside), dtype=np.float32)
    K_map = np.zeros(hp.nside2npix(nside), dtype=np.float32)
    n_map = np.zeros(hp.nside2npix(nside), dtype=np.float32)
    
    for catalog in shear_catalogs:
        if verbose: print("Reading {}.".format(catalog))
        ra, dec, e1, e2, w, m = prepare_catalog(catalog, column_names, 
                                                c_correction, m_correction, 
                                                z_min, z_max, selections, hdu_idx)
        # Depending on the definition of the coordinate system and ellipticities, one 
        # ellipticity component might have to be flipped.
        if flip_e1:
            e1 = -e1

        if convert_to_galactic:
            if verbose: print("Converting to Galactic coordinates.")
            lon, lat, e1, e2 = convert_shear_to_galactic_coordinates(ra, dec, e1, e2)
            # Find pixel indices of the catalog objects
            pixel_idx = hp.ang2pix(nside, lon, lat, lonlat=True)
        else:
            pixel_idx = hp.ang2pix(nside, ra, dec, lonlat=True)

        # In case we want random shear maps
        if shear_randoms:
            n = e1.size
            alpha = pi*np.random.rand(n)
            e = np.sqrt(e1**2 + e2**2)
            e1 = np.cos(2.0*alpha)*e
            e2 = np.sin(2.0*alpha)*e

        # Add the ellipticities to their corresponding healpix pixel.
        # at() is a lot faster than looping over indicies!
        np.add.at(e1_map, pixel_idx, e1*w)
        np.add.at(e2_map, pixel_idx, e2*w)
        np.add.at(w_map, pixel_idx, w)
        np.add.at(n_map, pixel_idx, 1)
        np.add.at(K_map, pixel_idx, w*(1+m))


    # Take the average by dividing by the total weight
    weight_mask = w_map > 0
    e1_map[weight_mask] /= K_map[weight_mask]
    e2_map[weight_mask] /= K_map[weight_mask]

    if verbose: print("Masking and writing to "+('/'.join(out_filenames['triplet'].split('/')[:-1])))

    # Masking the maps
    e1_map[~weight_mask] = masked_value
    e2_map[~weight_mask] = masked_value

    #w_map = hp.ma(w_map)
    #w_map.mask = ~weight_mask

    #weight_mask_map = hp.ma(np.ones(hp.nside2npix(nside), dtype=np.float32), copy=False)
    #weight_mask_map.mask = ~weight_mask
    
    weight_mask_map = (weight_mask!=0)*1.0
    
    nbar = np.mean(n_map[n_map!=0])
    rho_map = (n_map - nbar)/nbar
    rho_map[~weight_mask] = masked_value
    
    #T = np.ones(hp.nside2npix(nside), dtype=np.float32)*masked_value

    counts_mask_map = np.ones(hp.nside2npix(nside), dtype=np.float32)*masked_value
    counts_mask_map[weight_mask] = 1
    # Give PolSpice one non-zero pixel to work with
    #T = w_map/np.average(w_map[weight_mask])-1#np.random.randn(np.count_nonzero(weight_mask))*0.001
    #T[~weight_mask] = masked_value
    #counts_mask_map[0] = 1

    coord = "G" if convert_to_galactic else "C"

    # Writing to disk
    if partial_maps:
        write_partial_polarization_file([rho_map, e1_map, e2_map], nside=nside, filename=out_filenames["triplet"], mask_value=masked_value, coord=coord)
        hp.write_map(out_filenames["singlet_mask"], counts_mask_map, partial=True, coord=coord, fits_IDL=False, overwrite=True)
        hp.write_map(out_filenames["doublet_mask"], weight_mask_map, partial=True, coord=coord, fits_IDL=False, overwrite=True)
        hp.write_map(out_filenames["doublet_weight"], w_map, partial=True, coord=coord, fits_IDL=False, overwrite=True)
    else:
        hp.write_map(out_filenames["triplet"], [rho_map, e1_map, e2_map], partial=False, coord=coord, fits_IDL=False, overwrite=True)
        hp.write_map(out_filenames["singlet_mask"], counts_mask_map, partial=False, coord=coord, fits_IDL=False, overwrite=True)
        hp.write_map(out_filenames["doublet_mask"], weight_mask_map, partial=False, coord=coord, fits_IDL=False, overwrite=True)
        hp.write_map(out_filenames["doublet_weight"], w_map, partial=False, coord=coord, fits_IDL=False, overwrite=True)

        
def create_foreground_healpix_triplet(maps, masks, out_filenames, nside, 
                                      coord_in="G", coord_out="C", 
                                      verbose=True, footprint_file=None):
    """Convert HEALPix map in Galactic coordinates to celestial coordinates
    and produce triplet map to use with PolSpice in polarization mode.abs

    Required arguments:
    maps                Dictionary with entries "file" and "field".
    masks               List of dictionaries describing the masks. The 
                        dictionaries have the same format as maps.
    out_filenames       Dictinary of file names of the output healpix maps. 
                        Required entries are "triplet", "singlet_mask", and
                        "doublet_mask"..
    nside               HEALPix nside parameter.
    
    Optional arguments:
    coord_in            Coordinate frame of the input map. Either "G" or "C".
    coord_out           Coordinate frame of the output map. Either "G" or "C".
    footprint_file      HEALPix map containing the footprint of the lensing map.
    verbose             Verbose output (default: True).
    """
    if verbose: print("Reading file.")
    # Read healpix map
    m = hp.read_map(maps["file"], field=maps["field"])
    
    if coord_in == "G" and coord_out == "C":
        # Get coordinates of pixels
        theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
        # Transform Galactic coordinates to celestial coordinates
        cel_coords = astropy.coordinates.ICRS(ra=phi*units.rad, dec=(-theta+pi/2)*units.rad)
        gal_coords = cel_coords.transform_to(astropy.coordinates.Galactic)
        # Look up map values at the transformed coordinates, using interpolation
        if verbose: print("Transforming to celestial coordinates.")
        m = hp.get_interp_val(m, -gal_coords.b.rad+pi/2, gal_coords.l.rad)
    elif coord_in == "C" and coord_out == "G":
        # Get coordinates of pixels
        theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
        # Transform Galactic coordinates to celestial coordinates
        gal_coords = astropy.coordinates.Galactic(l=phi*units.rad, b=(-theta+pi/2)*units.rad)
        cel_coords = gal_coords.transform_to(astropy.coordinates.ICRS)
        # Look up map values at the transformed coordinates, using interpolation
        if verbose: print("Transforming to galactic coordinates.")
        m = hp.get_interp_val(m, -cel_coords.dec.rad+pi/2, cel_coords.ra.rad)

    if footprint_file is not None:
        footprint = hp.read_map(footprint_file).astype(bool)
    else:
        footprint = None
        
    #Apply masks
    if masks != None:
        if verbose: print("Masking.")
        mask = np.ones(hp.nside2npix(nside), dtype=bool)
        for mask_config in masks:
            mask = np.logical_and(mask, hp.read_map(mask_config["file"], field=mask_config["field"]))
        
        if coord_in == "G" and coord_out == "C":
            mask = mask[hp.ang2pix(nside, -gal_coords.b.rad+pi/2, gal_coords.l.rad)]
        elif coord_in == "C" and coord_out == "G":
            mask = mask[hp.ang2pix(nside, -cel_coords.dec.rad+pi/2, cel_coords.ra.rad)]

        doublet_mask = np.zeros_like(mask).astype(bool)
        if footprint is not None:
            doublet_mask[footprint] = 1

        hp.write_map(out_filenames["doublet_mask"], doublet_mask, coord=coord_out, fits_IDL=False, overwrite=True)
        hp.write_map(out_filenames["singlet_mask"], mask, coord=coord_out, fits_IDL=False, overwrite=True)
        m = hp.ma(m)
        m.mask = np.logical_not(mask)

    QU = np.random.random(size=m.shape)#np.zeros_like(m)
    if footprint is not None:
        QU[footprint] = np.random.randn(np.count_nonzero(footprint))*0.001
    
    if verbose: print("Writing to file.")
    hp.write_map(out_filenames["triplet"], [m, QU, QU], coord=coord_out, fits_IDL=False, overwrite=True)

