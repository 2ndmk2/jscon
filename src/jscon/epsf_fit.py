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


def main(image, x_select, y_select,  ext_size= 25, n_oversample = 2, n_maxiter=10, norm_radius = 5.5, recentering_boxsize =7):
    """
        epsf fitting for image given stellar positions (x_select, y_select). 

 
        Params:
            image: 2D image
            x_select: Array for x-positions for stars in pixel coordinate
            y_select: Array for y-positions for stars in pixel coordinate
            ext_size: Cutout size for epsf models & centroid fiting ("ext_size" * "ext_size" pixels)
            n_oversample: Oversampling rate for epsf building
            n_maxiter: Maximum iteration for epsf fitting & centroid fitting
            norm_radius: The radius inside which the ePSF is normalized by the sum over undersampled integer pixel values inside a circular aperture??
            recentering_boxsize: The size (in pixels) of the box used to calculate the centroid of the ePSF during each build iteration. If a single integer number is provided, \
                then a square box will be used. If two values are provided, then they should be in (ny, nx) order.
        Returns:
            dx: Array of difference of x-positions between estimated and reference positions
            dy: Array of difference of x-positions between estimated and reference positions
            epsf_image: 2D epsf image
    """
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
    epsf_image = epsf.data 

    return dx, dy, epsf_image

