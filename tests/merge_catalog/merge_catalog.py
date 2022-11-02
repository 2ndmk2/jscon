import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astroML.datasets import fetch_imaging_sample, fetch_sdss_S82standards
from astroML.crossmatch import crossmatch_angular

gns_catalog_path = "/Volumes/G-DRIVE/jasmine/gns_survey/J_A+A_631_A20/gns_catalogs.csv"
ohsawa_catalog_path = "/Volumes/G-DRIVE/jasmine/working_dir/ohsawa_catalog/all_table.csv"
gns_data = pd.read_csv(gns_catalog_path)
ohsawa_data = pd.read_csv(ohsawa_catalog_path)

## Divid gns_catalog into bright and faint stars
## We also extract bright stars from ohsawa catalog

bright_threshold = 13
bright_threshold_ohsawa = 13.5
gns_bright_mask = gns_data["Jmag"]<bright_threshold
gns_faint = gns_data[gns_bright_mask==False]
gns_bright = gns_data[gns_bright_mask]
ohsawa_bright =  ohsawa_data [ohsawa_data ["phot_j_mag"]<bright_threshold_ohsawa]


### Crossmatch bright gns catalog with ohsawa catalog
imX = np.empty((len(gns_data ["ra"]), 2), dtype=np.float64)
imX[:, 0] =gns_data ['ra']
imX[:, 1] = gns_data ['dec']

imX_bright = np.empty((len(gns_bright['ra']), 2), dtype=np.float64)
imX_bright[:, 0] = gns_bright['ra']
imX_bright[:, 1] = gns_bright['dec']

stX = np.empty((len(ohsawa_bright ["ra"]), 2), dtype=np.float64)
stX[:, 0] =ohsawa_bright ['ra']
stX[:, 1] = ohsawa_bright ['dec']

## Stars should be near to one of gns stars at least 5 arcsec
max_radius = 5 / 3600  # 5 arcsec
max_radius_bright = 0.5 / 3600  # 5 arcsec
dist, ind = crossmatch_angular(stX, imX, max_radius)
dist_bright, ind_bright = crossmatch_angular(stX, imX_bright, max_radius_bright )

### Choose bright stars only in ohsawa catalog
crossmatch_dist_thr = .3 / 3600 # .2 arcsec
match = ~np.isinf(dist) ## ohsawa bright stars near to GNS starss
match_bright = ~np.isinf(dist_bright) ## There are stars both in ohasaw & gns catalog
match_bright_only_ohsawa = np.isinf(dist_bright) ## Stars only in ohasawa catalog
mask_extra_stars = match * match_bright_only_ohsawa## There is no corresponding stars in gns catalog


### Merge ohsawa bright stars & gns catalog
ohsawa_data_bright_star = ohsawa_bright[mask_extra_stars]
ohsawa_data_bright_star = ohsawa_data_bright_star.rename(columns={"phot_j_mag":"Jmag","phot_h_mag":"Hmag",\
    "phot_ks_mag":"Ks_mag","phot_j_mag_error":"e_Jmag", "phot_h_mag_error":"e_Hmag",\
        "phot_ks_mag_error":"e_Ks_mag"} )
ohsawa_data_bright_star = ohsawa_data_bright_star [["ra", "dec", "Jmag", "Hmag", "Ks_mag","e_Jmag","e_Hmag","e_Ks_mag"]]
merged_df = pd.concat([ohsawa_data_bright_star , gns_data, ])

merged_df.to_csv("./gns+ohsawa_bright_stars.csv",)
