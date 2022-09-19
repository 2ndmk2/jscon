from jscon import parameter_load
import pandas as pd
import numpy as np
from jscon import make_image as mk
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import matplotlib.pyplot as plt

def make_bright_stars(gal_l_min, gal_l_max, gal_b_min, gal_b_max, gal_l, gal_b , \
                      hwmag, hwmag_max = 15, out_file = "../data/new_stellar_catalog.csv" ):
    del_l = gal_l_max - gal_l_min
    del_b = gal_b_max - gal_b_min    
    del_l_target = np.max(gal_l) - np.min(gal_l) 
    del_b_target = np.max(gal_b) - np.min(gal_b) 
    n_star =int( len(hwmag) *  (del_l/del_l_target )*  (del_b/del_b_target ))
    l_sample = np.random.rand(n_star) *del_l  + gal_l_min 
    b_sample = np.random.rand(n_star) *del_b  + gal_b_min 
    hwmag_sample = np.random.choice(hwmag, size= n_star)
    c_new = SkyCoord(l_sample, b_sample, frame="galactic", unit="deg")
    ra_new = c_new.icrs.ra.deg
    dec_new = c_new.icrs.dec.deg
    dic_new = {}
    dic_new["ra"] = ra_new
    dic_new["dec"] = dec_new
    dic_new["hwmag"] = hwmag_sample
    df = pd.DataFrame(dic_new)
    mask = df ["hwmag"] < 14.5
    df_select_bright= df [ mask ]
    df_select_bright.to_csv(out_file )
    return df_select_bright

def make_new_stars(gal_l_min, gal_l_max, gal_b_min, gal_b_max, gal_l, gal_b , hwmag, out_file = "../data/new_stellar_catalog.csv" ):
    del_l = gal_l_max - gal_l_min
    del_b = gal_b_max - gal_b_min    
    del_l_target = np.max(gal_l) - np.min(gal_l) 
    del_b_target = np.max(gal_b) - np.min(gal_b) 
    n_star =int( len(hwmag) *  (del_l/del_l_target )*  (del_b/del_b_target ))
    l_sample = np.random.rand(n_star) *del_l  + gal_l_min 
    b_sample = np.random.rand(n_star) *del_b  + gal_b_min 
    hwmag_sample = np.random.choice(hwmag, size= n_star)
    c_new = SkyCoord(l_sample, b_sample, frame="galactic", unit="deg")
    ra_new = c_new.icrs.ra.deg
    dec_new = c_new.icrs.dec.deg
    
    c = SkyCoord(ra_new, dec_new, frame="icrs", unit="deg")
    gal_l = c.galactic.l.deg
    gal_b = c.galactic.b.deg       
    dic_new = {}
    dic_new["ra"] = ra_new
    dic_new["dec"] = dec_new
    dic_new["hwmag"] = hwmag_sample
    dic_new["gal_l"] = gal_l
    dic_new["gal_b"] = gal_b

    df = pd.DataFrame(dic_new)
    df.to_csv(out_file)
    
    
    return df