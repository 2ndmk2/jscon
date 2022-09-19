import glob
import numpy as np
import matplotlib.pyplot as plt
from jscon import astrometry
import importlib 
importlib.reload(astrometry)
from jscon import make_image as mk
from astropy.coordinates import SkyCoord  # High-level coordinates
import astropy.units as u
import os
from jscon import parameter_load
def moving_average(x, w, time_arr):
    time_arr_conv = time_arr[:len(x) - w+1] + (w/2) * (time_arr[1] - time_arr[0])
    return np.convolve(x, np.ones(w), 'valid') / w, time_arr_conv 

def analyze_result(out_folder = "../out", para_file = "../params/para_yajiri.dat", n_star = 20, wd_time= 150, plot = True):

    dic_params = parameter_load.load_params(para_file)
    pix_arcsec = dic_params["pix_arcsec"]                  
    astro_info = np.load(os.path.join(out_folder, "input_astro_info.npz"))
    files = sorted(glob.glob(os.path.join(out_folder, "displace*.npz") ))

    mask_targets = astro_info["mask_targets"]
    ra = astro_info["ra"][mask_targets]
    dec = astro_info["dec"][mask_targets]
    pm_ra = astro_info["pm_ra"][mask_targets]
    pm_dec = astro_info["pm_dec"][mask_targets]
    distance = astro_info["distance"][mask_targets]
    time_arr= astro_info["time"]
    time_ref = astro_info["time_ref"]
    files = []
    for time_now in time_arr:
        out_file = os.path.join( out_folder , "displace_%.5f.npz" % (time_now))
        files.append(out_file)

    dx_arr = []
    dy_arr = []
    for file in files:
        data = np.load(file)
        dx_arr.append(data["dx"] * pix_arcsec * 1000)
        dy_arr.append(data["dy"] * pix_arcsec* 1000)  
    dx_arr = np.array(dx_arr)
    dy_arr = np.array(dy_arr)
    ra_delta, dec_delta= astrometry.compute_delta_ra_dec_all_phases(ra, dec, pm_ra, pm_dec, distance, time_arr, time_ref)
    
    dx_dif = []
    dy_dif = []
    print("numstar: %d" % (len(dx_arr[0])))
    
    if n_star== -1:
        n_star = len(dx_arr[0])
        
    for idx in range(0, n_star ):


        ra_delta_now = ra_delta[:,idx] #/pix_arcsec
        dec_delta_now = dec_delta[:,idx] #/pix_arcsec
        ra_shift_mean = np.mean( dx_arr[:,idx]-ra_delta_now )
        dec_shift_mean = np.mean( dy_arr[:,idx]-dec_delta_now )
        dx_dif.append(np.std(dec_shift_mean +dec_delta_now - dx_arr[:,idx]))
        dy_dif.append(np.std(dec_shift_mean +dec_delta_now - dx_arr[:,idx]))
        if plot :
            print("===== star %d ====" % idx)
            
            dx_ave, time_arr_conv = moving_average(dx_arr[:,idx], wd_time, time_arr)
            dy_ave, time_arr_conv = moving_average(dy_arr[:,idx], wd_time, time_arr)

            fig = plt.figure(figsize = (10,8))
            plt.plot(time_arr, dx_arr[:,idx],  label="data")
            plt.plot(time_arr, ra_shift_mean + ra_delta_now, label="model", lw = 5)
            plt.plot(time_arr_conv ,  dx_ave,  label="bin data")
            plt.xlabel("time [year]", fontsize = 20)
            plt.ylabel("x pixel [mas]", fontsize = 20)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=16)
            plt.legend(fontsize = 20)
            plt.show()

            ylim_wd, ylim_min, ylim_max= 0.3 * (np.max(dx_ave) - np.min(dx_ave)),  np.min(dx_ave),  np.max(dx_ave)
            fig = plt.figure(figsize = (10,8))
            #plt.plot(time_arr, dx_arr[:,idx],  label="data")
            plt.plot(time_arr, ra_shift_mean + ra_delta_now, label="model", lw = 5)
            plt.plot(time_arr_conv ,  dx_ave,  label="bin data")
            plt.xlabel("time [year]", fontsize = 20)
            plt.ylabel("x pixel [mas]", fontsize = 20)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=16)
            plt.legend(fontsize = 20)
            plt.ylim(ylim_min - ylim_wd, ylim_max + ylim_wd,)
            plt.show()

            ylim_wd, ylim_min, ylim_max= 0.3 * (np.max(dy_ave) - np.min(dy_ave)),  np.min(dy_ave), np.max(dy_ave)
            fig = plt.figure(figsize = (10,8))
            plt.plot(time_arr, dy_arr[:,idx],  label="data")
            plt.plot(time_arr, dec_shift_mean +dec_delta_now, label="model", lw = 5)
            plt.plot(time_arr_conv ,  dy_ave,  label="data")
            plt.xlabel("time [year]", fontsize = 20)
            plt.ylabel("y pixel [mas]", fontsize = 20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=12)
            plt.legend(fontsize = 20)
            plt.show()

            fig = plt.figure(figsize = (10,8))
            #plt.plot(time_arr, dy_arr[:,idx],  label="data")
            plt.plot(time_arr, dec_shift_mean +dec_delta_now, label="model", lw = 5)
            plt.plot(time_arr_conv ,  dy_ave,  label="data")
            plt.xlabel("time [year]", fontsize = 20)
            plt.ylabel("y pixel [mas]", fontsize = 20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=12)
            plt.ylim(ylim_min - ylim_wd, ylim_max + ylim_wd,)
            plt.legend(fontsize = 20)
            plt.show()
    return dx_dif, dy_dif, dx_arr, dy_arr
