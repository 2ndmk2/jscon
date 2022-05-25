import numpy as np
import matplotlib.pyplot as plt
import copy, warnings
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pf
from astropy.modeling import models, fitting
from astropy.nddata import NDData
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.visualization import simple_norm
from photutils.background import MMMBackground
from photutils.detection import find_peaks
from photutils.psf import EPSFBuilder, extract_stars


def main(image, x_target, y_target, lim_vals, mask_size= 25, ext_size= 9, n_oversample = 2, n_maxiter=10, norm_radius = 5.5, recentering_boxsize =7):

    x_min, x_max, y_min, y_max = lim_vals[0], lim_vals[1], lim_vals[2], lim_vals[3]
    x_target_plate = x_target- x_min
    y_target_plate = y_target- y_min

    mask_size = 25
    hsize = (mask_size - 1) / 2


    mask_region = ((x_target_plate > hsize) & (x_target_plate < (image.shape[1] -1 - hsize)) & (y_target_plate > hsize) & (y_target_plate < (image.shape[0] -1 - hsize)))
    x_select = x_target_plate[mask_region]
    y_select = y_target_plate[mask_region]

    stars_tbl = Table()
    stars_tbl['x'] = x_select
    stars_tbl['y'] = y_select

    mean_val, median_val, std_val = sigma_clipped_stats(image, sigma=2.)  
    image_copy = copy.deepcopy(image)
    image -= mean_val

    ###
    nddata = NDData(data=image)
    stars = extract_stars(nddata, stars_tbl, size=ext_size)

    #PSF 
    epsf_builder = EPSFBuilder(oversampling=n_oversample,
                                   maxiters=n_maxiter, progress_bar=False,
                                   norm_radius=norm_radius, recentering_boxsize=recentering_boxsize,
                                   center_accuracy=0.0001)

    epsf, fitted_stars = epsf_builder(stars)
    pos_stars = fitted_stars.center_flat
    dx= pos_stars[:,0] - x_select
    dy= pos_stars[:,1] - y_select  
    return dx, dy  

