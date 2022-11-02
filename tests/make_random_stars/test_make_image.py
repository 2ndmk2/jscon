import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from jscon import make_image as mk_image
from jscon import make_stars as mk_star

file_random_stars = "./random_stars.csv"
make_image = True

npix = 2000 
pix_arcsec = 0.4 ## pixel scale
sky_dir_l, sky_dir_b = 0, 0

## random star dataframe
df_random_stars = pd.read_csv(file_random_stars)

# Make image based on 
sigma_to_pix = 1 ## Gaussian beam FWHM in unit of pixel. Change to realistic value
gauss_conv = mk_image.compute_espf_for_gaussian_psf(sigma_pix = sigma_to_pix)
pixel_targets_random = mk_image.make_pixel_coordinates(df_random_stars["gal_l"], df_random_stars["gal_b"], sky_dir_l, sky_dir_b, pix_arcsec)
image_random, mask_included = mk_image.make_image(pixel_targets_random, df_random_stars["hwmag"], gauss_conv, npix,npix)
percen = np.percentile(image_random, (1, 99.9))
fig = plt.figure(figsize = (20, 20))
plt.imshow(image_random , vmin = percen[0], vmax = percen[1])
plt.colorbar()
plt.show()