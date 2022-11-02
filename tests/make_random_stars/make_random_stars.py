import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from jscon import make_image as mk_image
from jscon import make_stars as mk_star

file_random_stars = "./random_stars.csv"

# Load catalog
catalog_name = "/Volumes/G-DRIVE/jasmine/working_dir/2022_1027/gns+ohsawa_bright_stars.csv"
catalog = mk_image.get_catalog_info_for_making_stars(catalog_name)

# Latititude Shift from (0, 360) -> (-180, 180)
catalog_shifted = catalog
mask_shifted = catalog_shifted["gal_l"]>180
catalog_shifted["gal_l"][mask_shifted] -=360

# Stellar distribution use for makeing random stars
l_min_choose = 0.18
l_max_choose = 0.24
b_min_choose = 0.15
b_max_choose = 0.17
mask_stars_for_learn = (catalog_shifted["gal_l"]>l_min_choose) * (catalog_shifted["gal_l"]<l_max_choose)* (catalog_shifted["gal_b"]>b_min_choose) * (catalog_shifted["gal_b"]<b_max_choose)
catalog_stars_for_learn = catalog_shifted[mask_stars_for_learn]

# Produce random stars with the same stellar density as input distribution
npix = 2000 ## Choose stars within (0, npix) * (0, npix)
pix_arcsec = 0.4 ## pixel scale
wd_for_image = 2000 * 0.4/3600 #deg

## random star dataframe
df_random_stars = mk_star.make_new_stars( catalog_stars_for_learn["gal_l"],catalog_stars_for_learn["gal_b"] ,catalog_stars_for_learn["hwmag"], 0, wd_for_image, 0,wd_for_image)
df_random_stars.to_csv(file_random_stars)
