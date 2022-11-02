import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import numpy as np
import astropy.units as u
import scipy.integrate as integrate
from scipy import interpolate

arc_rad = 4.84814e-6


def gaussian_1d(x, sigma_x):
    """1d gaussian function

    Params:
        x: x-pos
        sigma_x: gaussian sigma
    Returns:
        value: value for gaussian 
    """
    value = (1 / (np.sqrt(2 * np.pi) * sigma_x)) * np.exp(-(x**2) /
                                                          (2 * sigma_x**2))
    return value


def pixel_1d_gaussian(x_cen, dx, sigma_x):
    """1d gaussian function convolved with pixel response function with pulse shape

    Params:
        x_cen: center for pixel 
        sigma_x: dx width for pixel
        sigma_x: gaussian sigma
    Returns:
        value: value for convolved gaussian 
    """

    result_x = integrate.quad(lambda x: gaussian_1d(x, sigma_x), x_cen - dx,
                              x_cen + dx)
    value = result_x[0]
    return value


def create_wcs(skydir,
               coordsys='CEL',
               projection='TAN',
               cdelt=1.0,
               crpix=1.,
               naxis=2):
    """Create a WCS object.
    Taken from https://github.com/fermiPy/dmsky/blob/master/dmsky/utils/wcs.py

    Params:
        skydir: Sky coordinate of the WCS reference point.
        coordsys : str
        projection : str
        cdelt : In the first case the same value is used for x and y axes
        crpix : In the first case the same value is used for x and y axes
        naxis : Number of dimensions of the projection.
        energies : Array of energies that defines the third dimension if naxis=3.
    Returns:
        w: WCS object
    """

    w = WCS(naxis=naxis)

    if coordsys == 'CEL':
        w.wcs.ctype[0] = 'RA---%s' % (projection)
        w.wcs.ctype[1] = 'DEC--%s' % (projection)
        w.wcs.crval[0] = skydir.icrs.ra.deg
        w.wcs.crval[1] = skydir.icrs.dec.deg
    elif coordsys == 'GAL':
        w.wcs.ctype[0] = 'GLON-%s' % (projection)
        w.wcs.ctype[1] = 'GLAT-%s' % (projection)
        w.wcs.crval[0] = skydir.galactic.l.deg
        w.wcs.crval[1] = skydir.galactic.b.deg
    else:
        raise Exception('Unrecognized coordinate system.')

    try:
        w.wcs.crpix[0] = crpix[0]
        w.wcs.crpix[1] = crpix[1]
    except:
        w.wcs.crpix[0] = crpix
        w.wcs.crpix[1] = crpix

    try:
        w.wcs.cdelt[0] = cdelt[0]
        w.wcs.cdelt[1] = cdelt[1]
    except:
        w.wcs.cdelt[0] = -cdelt
        w.wcs.cdelt[1] = cdelt

    return w


def make_pixel_coordinates(gal_l,
                           gal_b,
                           sky_dir_l=359.85,
                           sky_dir_b=0.54,
                           pix_arcsec=0.4,
                           x_offset=50,
                           y_offset=50):
    """
    Params:
        gal_l: array for galactic longitude coordinate
        gal_b: array for galactic latitude coordinate
        skydir_l: Sky position for galactic longitude coordinate at (0pix, 0pix)
        skydir_b: Sky position for galactic latitude coordinate at (0pix, 0pix)

    """

    arcsec_deg = 0.00027777777777778  ##deg to radian
    cdelt = [pix_arcsec * arcsec_deg, pix_arcsec * arcsec_deg]
    skydir = SkyCoord(sky_dir_l * u.deg, sky_dir_b * u.deg, frame="galactic")
    wcs = create_wcs(skydir, "GAL", crpix=[0, 0], cdelt=cdelt)
    pixel_targets = wcs.all_world2pix(gal_l, gal_b, 1)
    pixel_targets[0] += x_offset
    pixel_targets[1] += y_offset
    return pixel_targets



def get_catalog_info_for_making_stars(file_name):
    """ Get dataframe from file with name of "file_name". We only take stars with J & H mag available

    Params:
        file_name: file_name for csv file
    Returns:
        df: data
    """

    df= pd.read_csv(file_name)
    mask = (df["Hmag"] < 30) *  (df["Jmag"] < 30)
    hwmag = 0.7829 * df["Jmag"] + 0.2171 * df["Hmag"] -0.0323* (df["Jmag"] -df["Hmag"])**2  
    c = SkyCoord(df["ra"].values, df["dec"].values, frame="icrs", unit="deg")
    gal_l = c.galactic.l.deg 
    gal_b = c.galactic.b.deg 
    df["hwmag"] = hwmag 
    df["gal_l"] = gal_l
    df["gal_b"] = gal_b
    df = df[mask]
    return df

def get_gal_l_gal_b_hwmag(df, mag_lim = 14.5, minus_l = True):
    hwmag_targets = df["hwmag"] 
    c = SkyCoord(df["ra"].values, df["dec"].values, frame="icrs", unit="deg")
    if minus_l:
        gal_l = c.galactic.l.deg - 180
    else:
        gal_l = c.galactic.l.deg
    gal_b = c.galactic.b.deg
    hwmag = df["hwmag"]
    return gal_l, gal_b, hwmag, hwmag_targets


def get_catalog_info(file_name):
    """ Get dataframe from file with name of "file_name". We only take stars with J & H mag available

    Params:
        file_name: file_name for csv file
    Returns:
        df: data
    """

    df = pd.read_csv(file_name)
    #mask = (df["Hmag"] < 30) *  (df["Jmag"] < 30)
    #hwmag = 0.7829 * df["Jmag"] + 0.2171 * df["Hmag"] -0.0323* (df["Jmag"] -df["Hmag"])**2
    #df["hwmag"] = hwmag
    #df = df[mask]
    return df


def compute_espf_for_gaussian_psf(sigma_pix, width=100, N_width=10000):
    """ Construct epsf for 1d gaussian psf
    
    Params:
        sigma_pix: std for beam in unit of pix
        width: The psf is constructed on "" -width [pix] < x < width [pix], -width [pix] < y < width [pix] "" 
        N_width: -width [pix] < x < width [pix] is divided into N_width sections
    Returns:
        sigma_pix: combined std for beam 
    """
    x_cen_arr = np.linspace(-width, width, N_width)
    y_val_arr = []
    for x_cen in x_cen_arr:
        val = pixel_1d_gaussian(x_cen, .5, sigma_pix)
        y_val_arr.append(val)
    gauss_conv = interpolate.interp1d(x_cen_arr, y_val_arr)
    return gauss_conv


def compute_sigma(lambda_now, diameter, sigma_ace, pix_arcsec):
    """ Compute sigma for beam based on psf width (Airy function) and ace variance. 

    Params:
        lambda_now: wavelength (m)
        diameter: telescope diameter (m)
        pix_arcsec: pixscale in unit of arcsec (arcsec)
    Returns:
        sigma_pix: combined std for beam (pix)
    """
    sigma_psf_arcsec_val = (0.512 / 1.177410) * (lambda_now /
                                                 diameter) / arc_rad
    sigma_tot = np.sqrt(sigma_psf_arcsec_val**2 + sigma_ace**2)
    sigma_pix = sigma_tot / pix_arcsec  #unit of pix

    return sigma_pix


def make_image(pixel_targets,
               hwmag_targets,
               gauss_conv,
               x_max=1400,
               y_max=700,
               wd_margin=30,
               wd_psf=20):
    """ Make image frame with size of (0:xmax, 0:ymax ) based on stellar inputs

    Params:
        pixel_targets: Array for stellar positions (2*N_star, unit of pix*pix)
        hwmag_targets: Array for Hw magnitude
        gauss_conv: ePSF function. Convert (x, y) to flux
        x_min: Left side of x-cooridnate 
        x_max: Right side of x-cooridnate 
        y_min: Left side of x-cooridnate 
        y_max: Right side of y-cooridnate 
    Returns:
        image: 2d image
    """
    x_coord = np.arange(x_max)
    y_coord = np.arange(y_max)
    x_id = np.arange(len(x_coord))
    y_id = np.arange(len(y_coord))
    image = np.zeros((len(x_coord), len(y_coord)))
    mask_included = []

    for i in (range(len(pixel_targets[0]))):
        x_cen = pixel_targets[0][i]
        y_cen = pixel_targets[1][i]
        if wd_margin < x_cen and x_cen < x_max - wd_margin and wd_margin < y_cen and y_cen < y_max - wd_margin:
            x_dif = np.abs(x_coord - x_cen)
            y_dif = np.abs(y_coord - y_cen)
            mask_x = x_dif < wd_psf
            mask_y = y_dif < wd_psf
            x_min_id = np.min(x_id[mask_x])
            x_max_id = np.max(x_id[mask_x])
            y_min_id = np.min(y_id[mask_y])
            y_max_id = np.max(y_id[mask_y])
            x_f = gauss_conv(x_dif[mask_x])
            y_f = gauss_conv(y_dif[mask_y])
            c = 20 * 10**(-0.4 * (hwmag_targets[i] - 20)) * np.outer(x_f, y_f)
            image[x_min_id:x_max_id + 1, y_min_id:y_max_id + 1] += c
            mask_included.append(True)
        else:
            mask_included.append(False)
    mask_included = np.array(mask_included)
    image = image.T
    return image, mask_included
