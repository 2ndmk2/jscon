import numpy as np
import astropy.coordinates
from astropy import units as u
from astropy import constants
from astropy.time import Time

earth_sun_mass_ratio = (constants.M_earth/constants.M_sun).value

def barycentricPosition(time):

    """ Taken from astromet (./astromet/tracks.py)
    https://github.com/zpenoyre/astromet.py

    Compute barycentric position for earth
    Params:
        time: Array or float for time (year)

    Returns:
        earth_pos: Array for earth position
    """

    pos = astropy.coordinates.get_body_barycentric('earth', astropy.time.Time(time, format='jyear'))
    xs = pos.x.value  # all in AU
    ys = pos.y.value
    zs = pos.z.value
    l2corr = 1+np.power(earth_sun_mass_ratio/3, 1/3)
    earth_pos = l2corr*np.vstack([xs, ys, zs]).T
    return earth_pos

def compute_delta_ra_dec_one_phase(ra, dec, pm_ra, pm_dec, distance, time_now, time_origin):
    """ Compute delta (ra, dec) for objects considering "parallax" & "proper" motion

    Params:
        ra: Array for ra for objects (rad)
        dec: Array for dec for objects (rad)
        pm_ra: Array for proper motion in ra for objects (mas/yr)
        pm_dec: Array for proper motion in dec for objects (mas/yr)
        distance: Array for distances to objects (kpc)
        time_now: Time for observation (year, e.g. 2020)
        time_origin: Time origin for observation. Proper motion =0 if time_now = time_origin. (year, e.g. 2024)

    Returns:
        ra_delta: Shift in ra direction for objects (mas)
        dec_delta: Shift in dec direction for objects (mas)

    """

    
    time_delta = time_now - time_origin
    bs = barycentricPosition(time_now)
    ra_delta_for_pm = time_delta * pm_ra
    dec_delta_for_pm = time_delta * pm_dec

    ## vector toward object
    p0 = np.array([-np.sin(ra), np.cos(ra), np.zeros(len(ra))])
    q0 = np.array([-np.cos(ra)*np.sin(dec), -np.sin(ra)*np.sin(dec), np.cos(dec)])
    ra_delta_parallax = -np.dot(p0.T, bs.T) #*(1/ distance) # parallax (ra component)
    dec_delta_parallax = -np.dot(q0.T, bs.T) #*(1/ distance) # parallax (dec component)
    ra_delta_parallax = ra_delta_parallax[:,0] *(1/ distance)
    dec_delta_parallax = dec_delta_parallax[:,0] *(1/ distance)
    ra_delta = ra_delta_for_pm  + ra_delta_parallax 
    dec_delta = dec_delta_for_pm  + dec_delta_parallax 
    return  ra_delta, dec_delta


def compute_delta_ra_dec_all_phases(ra, dec, pm_ra, pm_dec, distance, time_arr, time_origin):
    """ Compute delta (ra, dec) for objects considering "parallax" & "proper" motion

    Params:
        ra: Array for ra for objects (rad)
        dec: Array for dec for objects (rad)
        pm_ra: Array for proper motion in ra for objects (mas/yr)
        pm_dec: Array for proper motion in dec for objects (mas/yr)
        distance: Array for distances to objects (kpc)
        time_now: Time for observation (year, e.g. 2020)
        time_origin: Time origin for observation. Proper motion =0 if time_now = time_origin. (year, e.g. 2024)

    Returns:
        ra_all: Array for shift in ra direction for objects (mas)
        dec_all: Array for shift in dec direction for objects (mas)

    """
    ra_all = []
    dec_all = []
    for time_now in time_arr:
    
        time_delta = time_now - time_origin
        bs = barycentricPosition(time_now)
        ra_delta_for_pm = time_delta * pm_ra
        dec_delta_for_pm = time_delta * pm_dec

        ## vector toward object
        p0 = np.array([-np.sin(ra), np.cos(ra), np.zeros(len(ra))])
        q0 = np.array([-np.cos(ra)*np.sin(dec), -np.sin(ra)*np.sin(dec), np.cos(dec)])
        ra_delta_parallax = -np.dot(p0.T, bs.T) #*(1/ distance) # parallax (ra component)
        dec_delta_parallax = -np.dot(q0.T, bs.T) #*(1/ distance) # parallax (dec component)
        ra_delta_parallax = ra_delta_parallax[:,0] *(1/ distance)
        dec_delta_parallax = dec_delta_parallax[:,0] *(1/ distance)
        ra_delta = ra_delta_for_pm  + ra_delta_parallax 
        dec_delta = dec_delta_for_pm  + dec_delta_parallax 
        ra_all.append(ra_delta)
        dec_all.append(dec_delta)
    ra_all = np.array(ra_all)
    dec_all = np.array(dec_all)
    return  ra_all, dec_all


def compute_delta_ra_dec_one_phase_for_check(ra, dec, pm_ra, pm_dec, distance, time_now, time_origin):
    """ Compute delta (ra, dec) for objects considering "parallax" & "proper" motion

    Params:
        ra: Array for ra for objects (rad)
        dec: Array for dec for objects (rad)
        pm_ra: Array for proper motion in ra for objects (mas/yr)
        pm_dec: Array for proper motion in dec for objects (mas/yr)
        distance: Array for distances to objects (kpc)
        time_now: Time for observation (year, e.g. 2020)
        time_origin: Time origin for observation. Proper motion =0 if time_now = time_origin. (year, e.g. 2024)

    Returns:
        ra_delta_parallax: Shift in ra direction due to parallax motion for objects (mas)
        dec_delta_parallax: Shift in dec direction due to parallax motion for objects (mas)
        ra_delta_for_pm: Shift in ra direction due to proper motion for objects (mas)
        dec_delta_for_pm: Shift in dec direction due to proper motion for objects (mas)

    """

    
    time_delta = time_now - time_origin
    bs = barycentricPosition(time_now)
    ra_delta_for_pm = time_delta * pm_ra
    dec_delta_for_pm = time_delta * pm_dec

    ## vector toward object
    p0 = np.array([-np.sin(ra), np.cos(ra), np.zeros(len(ra))])
    q0 = np.array([-np.cos(ra)*np.sin(dec), -np.sin(ra)*np.sin(dec), np.cos(dec)])
    ra_delta_parallax = -np.dot(p0.T, bs.T) #*(1/ distance) # parallax (ra component)
    dec_delta_parallax = -np.dot(q0.T, bs.T) #*(1/ distance) # parallax (dec component)
    ra_delta_parallax = ra_delta_parallax[:,0] *(1/ distance)
    dec_delta_parallax = dec_delta_parallax[:,0] *(1/ distance)
    return  ra_delta_parallax, dec_delta_parallax, ra_delta_for_pm, dec_delta_for_pm


def make_astrometry(ra, dec, pm_ra, pm_dec, distance):
    """ make dictionary for stellar information for making images
    
    Params:
        ra: Array for ra for objects (rad)
        dec: Array for dec for objects (rad)
        pm_ra: Array for proper motion in ra for objects (mas/yr)
        pm_dec: Array for proper motion in dec for objects (mas/yr)
        distance: Array for distances to objects (kpc)
        
     Returns:
        df: Dictionary for containing astrometric infomation
    """
        
    df = {}
    df["ra"] = ra
    df["dec"] = dec
    df["pm_ra"] = pm_ra
    df["pm_dec"] = pm_dec
    df["distance"] = distance
    return df

    
def make_proper_motions_distance(n_star, amp =5, dist_min = 6500, dist_var = 3000):
    """ make dictionary for stellar information for making images
    
    Params:
        n_star: Number of stars
        amp: Amplitude for proper motion (-0.5*amp ~ 0.5*amp) (mas/year()
        dist_min: Minimum for distance (pc)
        dist_var: distance (dist_min ~ dist_min + dist_var) (pc)

     Returns:
        pm_ra: Array for proper motion in ra for objects (mas/yr)
        pm_dec: Array for proper motion in dec for objects (mas/yr)
        distance: Array for distances to objects (kpc)
    """


    distance = (np.ones(n_star) * dist_min +  dist_var * np.random.rand(n_star))/1000.0 ## * u.kpc
    pm_ra = amp * (-.5 + np.ones(n_star) * np.random.rand(n_star) )   #* u.mas/u.yr
    pm_dec = amp * (-.5 + np.ones(n_star) * np.random.rand(n_star) )   #* u.mas/u.yr
    return pm_ra, pm_dec, distance