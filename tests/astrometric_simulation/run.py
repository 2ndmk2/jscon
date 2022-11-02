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
import make_image as mk
import importlib
import epsf_fit
from multiprocessing import Process
import os

def main_pool(i_core, pixel_targets, hwmag_targets, gauss_conv,  lim_vals, mask, rand_nums, displace_amp = 3, n_iter =2, out_folder ="./out/",  n_oversample = 2,  n_maxiter=5):
    dx_arr = []
    dy_arr = []

    for i_ter in range(n_iter):
        displace = 2 * displace_amp * rand_nums[i_core, i_ter] - displace_amp
        pixel_targets_mov = np.array([pixel_targets[0] + displace [0], pixel_targets[1] + displace[1]])
        image = mk.make_image(pixel_targets_mov, hwmag_targets, gauss_conv)
        x_target = pixel_targets_mov [0][mask]
        y_target = pixel_targets_mov [1][mask]
        dx, dy  = epsf_fit.main(image, x_target, y_target, lim_vals, n_oversample = n_oversample, n_maxiter=n_maxiter )
        dx_arr.append(dx)
        dy_arr.append(dy)
        out_file = os.path.join( out_folder , "%d_%d" % (i_core, i_ter))
        np.savez(out_file, dx = dx, dy = dy, displace= displace)
        
    return dx_arr, dy_arr

def pre_main(pixel_targets, hwmag_targets, gauss_conv,  lim_vals, mask, displace_amp = 3, n_iter =2, out_folder ="./out/", n_oversample = 2, n_maxiter=5):

    displace = 2 * displace_amp * np.random.rand(2) - displace_amp
    pixel_targets_mov = np.array([pixel_targets[0] + displace [0], pixel_targets[1] + displace[1]])
    image = mk.make_image(pixel_targets_mov, hwmag_targets, gauss_conv)
    x_target = pixel_targets_mov [0][mask]
    y_target = pixel_targets_mov [1][mask]
    dx, dy  = epsf_fit.main(image, x_target, y_target, lim_vals , n_oversample =  n_oversample , n_maxiter=n_maxiter)
    dr = (dx**2 + dy**2)**0.5
    mask_bad =  dr  < 0.05
    mask_new = np.copy(mask)
    print(len( mask_new[mask][mask_bad]))
    print(len( mask_new[mask]))
    mask_new[mask] *= mask_bad
    
    return dx, dy, mask_new


if __name__=="__main__":

    out_folder ="./out/"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        
    df = mk.get_catalog_info("ibnorth.csv")
    mask_hwmag = df["hwmag"] < 14.5
    hwmag_targets = df["hwmag"] 
    c = SkyCoord(df["ra"].values, df["dec"].values, frame="icrs", unit="deg")
    gal_l = c.galactic.l.deg
    gal_b = c.galactic.b.deg

    crpix = [0,0]
    arcsec_deg =  0.00027777777777778 ##
    sigma_x = .8 # pixel for gaussian
    sigma_y = .8 # pixel for gaussian

    ## convolved gaussian 
    x_cen_arr = np.linspace(-100, 100, 20000)
    y_val_arr = []

    for x_cen in x_cen_arr:
        val = mk.pixel_1d_gaussian(x_cen, .5, sigma_x)
        y_val_arr.append(val)
    gauss_conv = mk.interpolate.interp1d(x_cen_arr,  y_val_arr)

    ## make WCS & pixel values for targets
    cdelt =[0.4 * arcsec_deg , 0.4 * arcsec_deg]
    skydir =  SkyCoord(359.85* u.deg , 0.54* u.deg, frame="galactic")#, l=0, b=0.5 * u.deg)
    wcs = mk.create_wcs(skydir, "GAL", crpix = crpix, cdelt = cdelt)
    pixel_targets = wcs.all_world2pix(gal_l, gal_b, 1)
    lim_vals = [0, 1400, -100, 700]
    
    dx, dy, mask_new = pre_main(pixel_targets, hwmag_targets, gauss_conv,  lim_vals, mask_hwmag)
    
    process_list = []
    n_core = 32
    n_iter = 2
    
    rand_nums = np.random.rand(n_core, n_iter, 2 )
    for i in range(n_core):
        process = Process(
            target=main_pool,
            kwargs={"i_core":i, 'pixel_targets': pixel_targets, "hwmag_targets":hwmag_targets, \
                    "gauss_conv": gauss_conv,"n_iter":n_iter, "lim_vals":lim_vals, "mask": mask_new,"out_folder":out_folder, "rand_nums":rand_nums})
        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()

    #pool = multiprocessing.Pool(processes=4)
    #pool.map(main_pool, [(pixel_targets, hwmag_targets, gauss_conv,  lim_vals, mask_hwmag)])

    """
    displace_amp = 3
    displace = 2 * displace_amp * np.random.rand(2) - displace_amp
    pixel_targets_mov = np.array([pixel_targets[0] + displace [0], pixel_targets[1] + displace[1]])
    image = mk.make_image(pixel_targets_mov, hwmag_targets, gauss_conv)

    x_target = pixel_targets_mov [0][mask_hwmag ]
    y_target = pixel_targets_mov [1][mask_hwmag ]
    dx, dy  = epsf_fit.main(image, x_target, y_target, lim_vals )



    plt.scatter(dx, dy)
    dr = (dx**2 + dy**2)**0.5
    mask = dr < 0.05
    plt.scatter(dx[mask], dy[mask])
    plt.show()
    """



