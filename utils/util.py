#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Handle light curves for convexinv.

Query JPL/HORIZONS and obtain quantities below.
  'Date__(UT)__HR:HM', 'R.A._____(ICRF)_____DEC', 'R.A.__(a-apparent)__DEC',
  'hEcl-Lon', 'hEcl-Lat', 'r', 'rdot', 'delta', 'deldot'

Calculate x_sun, y_sun ,z_sun (sun cartesian coodinate from the object)
and x_earth, y_earth, z_earth (earth cartesian coordinate from the object).
"""
import os 
import subprocess
import numpy as np
import pandas as pd
from scipy import interpolate
import datetime
from astroquery.jplhorizons import Horizons
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy
from astropy.time import Time


def adderr(*args):
    """Calculate additional error.

    Parameters
    ----------
    args : array-like
        list of values

    Return
    ------
    err : float
        calculated error
    """
    err = np.sqrt(np.sum(np.square(args)))
    return err




def time_keeper(func):
    """
    Decorator to measure time.
    """
    # To take over docstring etc.
    @wraps(func)
    def wrapper(*args, **kargs):
        t0 = time.time()
        result = func(*args, **kargs)
        t1 = time.time()
        t_elapse = t1 - t0
        print(f"[time keeper] t_elapse = {t_elapse:.03f} s (func :{func.__name__})")
        return result
    return wrapper


def Ariadnetestdata():
    # The time is same with test_lcs_rel in DAMIT code, not the text on website.
    dic_temp = dict(
        t_jd_ltcor=[
            2438882.233275, 2438882.237650, 2438882.241400, 2438882.245817,
            2438882.249192, 2438882.254275, 2438882.256984, 2438882.261359,
            2438882.264442, 2438882.270525],
        flux=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        )
    # JD is light-time corrected
    df = pd.DataFrame(dic_temp.values(), index=dic_temp.keys()).T
    return df


def format4inv(df, jpleph, key_jd):
    """Format lightcurves for convex inversion.

    jpleph should be queried by hand using columns 1,2,18,19,20.

    Note: 
        Of course, this code doesn't work well if the data is outside jpleph
        since info. in missing range is extrapolated in the code.
        Additionally, this result is different from the result using .vectors
        (as in the format4inv_query). So I don't use this function anymore.

    Parameters
    ----------
    df : str
        dataframe of lightcurve 
    jpleph : str
        filename of ephemeris from JPL
    key_jd : str
        keyword of time in JD

    Return
    ------
    df_format : pandas.DataFrame
        formatted data
    """
    # Create result = [df_phot1, df_phot2, ...] with jpl info.
    # create jpl dataframe
    utc_list, ra_list, dec_list = [], [], []
    hecl_lon, hecl_lat, r_list, delta_list = [], [], [], []
    jd_list, mjd_list = [], []
    x_sun_list, y_sun_list, z_sun_list = [], [], []
    x_earth_list, y_earth_list, z_earth_list = [], [], []
    with open(jpleph, "r") as f:
        # skip header of jpl ephem 
        n_header = 76
        #n_use = 121
        #f = f.readlines()[n_header:n_header+n_use]
        f = f.readlines()[n_header:]
        for line in f:
          # ok
          utc = line[1:18]
          # Ignore m and A and ...
          other = line[22:]
          # Split by space
          other = other.split(" ")
          # Replace "\n" with ""
          other = [x.replace("\n", "") for x in other]
          other = [x for x in other if x != ""]

          ## Check!
          if len(other)!=18:
              continue

          utc_list.append(utc)
          ra    = f"{other[0]} {other[1]} {other[2]}"
          dec   = f"{other[3]} {other[4]} {other[5]}"
          beta  = f"{other[12]}"
          lam   = f"{other[13]}"
          r     = f"{other[14]}"
          delta = f"{other[16]}"

          ra_list.append(ra)
          dec_list.append(dec)
          hecl_lon.append(beta)
          hecl_lat.append(lam)
          r_list.append(r)
          delta_list.append(delta)

          # jd
          ## convert utc (2020-Nov-30 16:00) to '1999-01-01T00:00:00.123'
          utc = datetime.datetime.strptime(utc, "%Y-%b-%d %H:%M")
          utc = datetime.datetime.strftime(utc, "%Y-%m-%dT%H:%M:%S")
          t = Time(utc, format='isot', scale='utc')
          jd_list.append(t.jd)
          mjd_list.append(t.mjd)
 
          # the Sun x, y, z
          ## use cartisian method
          c_eclip = SkyCoord(
              beta, lam, r, frame="heliocentricmeanecliptic", 
              unit=(u.deg, u.deg, u.au)
              )
          #print("The Sun ========================================================")
          #print("eclip lon, lat, r from the Sun to the asteroid")
          #print(f"{c_eclip.lon}, {c_eclip.lat}, {c_eclip.distance}")

          c_eclip = c_eclip.cartesian
          ## minus means from object
          #print("converted cartesian")
          x_sun = -c_eclip.x.value
          y_sun = -c_eclip.y.value
          z_sun = -c_eclip.z.value
          #print(f"{x_sun}, {y_sun}, {z_sun}")
          #print("")

          #print("The Earth ======================================================")
          # not icrs (), but gcrs(Geocentric Celestial Reference System)
          c_radec = SkyCoord(
              ra=ra, dec=dec, distance=delta, frame="gcrs", unit=(u.hourangle, u.deg, u.au),
              )
          #print("radec polar")
          #print(f"{c_radec.ra}, {c_radec.dec}, {c_radec.distance}")
          c_radec_car = c_radec.cartesian
          #print("radec cartesian")
          #print(f"{c_radec_car.x}, {c_radec_car.y}, {c_radec_car.z}")

          c_g_eclip_mean = c_radec.transform_to(
              astropy.coordinates.builtin_frames.GeocentricMeanEcliptic)
          #print("g_eclip converted from radec polar")
          #print(f"{c_g_eclip_mean.lon}, {c_g_eclip_mean.lat}, {c_g_eclip_mean.distance}")
          c_g_eclip_mean = c_g_eclip_mean.cartesian
          x_earth = -c_g_eclip_mean.x.value
          y_earth = -c_g_eclip_mean.y.value
          z_earth = -c_g_eclip_mean.z.value

          x_sun_list.append(x_sun)
          y_sun_list.append(y_sun)
          z_sun_list.append(z_sun)
          x_earth_list.append(x_earth)
          y_earth_list.append(y_earth)
          z_earth_list.append(z_earth)

    dic_temp = dict(
        utc=utc_list, ra=ra_list, dec=dec_list,
        hecl_lon=hecl_lon, hecl_lat=hecl_lat, r=r_list, delta=delta_list,
        jd=jd_list, mjd=mjd_list,
        x_sun=x_sun_list, y_sun=y_sun_list, z_sun=z_sun_list,
        x_earth=x_earth_list, y_earth=y_earth_list, z_earth=z_earth_list
        )

    # JD is not light-time corrected
    df_temp = pd.DataFrame(dic_temp.values(), index=dic_temp.keys()).T
    
    # function to predict various values when observation
    f_sun_x = interpolate.interp1d(
        df_temp["jd"], df_temp["x_sun"], kind='linear', fill_value='extrapolate')
    f_sun_y = interpolate.interp1d(
        df_temp["jd"], df_temp["y_sun"], kind='linear', fill_value='extrapolate')
    f_sun_z = interpolate.interp1d(
        df_temp["jd"], df_temp["z_sun"], kind='linear', fill_value='extrapolate')
    f_earth_x = interpolate.interp1d(
        df_temp["jd"], df_temp["x_earth"], kind='linear', fill_value='extrapolate')
    f_earth_y = interpolate.interp1d(
        df_temp["jd"], df_temp["y_earth"], kind='linear', fill_value='extrapolate')
    f_earth_z = interpolate.interp1d(
        df_temp["jd"], df_temp["z_earth"], kind='linear', fill_value='extrapolate')
    
    # Read photometric csv
    # light-time corrected unlike ephemerides from JPL
    col = df.columns.tolist()

    # predict x, y, z of the Sun and the Earth 
    df["x_sun"]   = df[key_jd].map(f_sun_x)
    df["y_sun"]   = df[key_jd].map(f_sun_y)
    df["z_sun"]   = df[key_jd].map(f_sun_z)
    df["x_earth"] = df[key_jd].map(f_earth_x)
    df["y_earth"] = df[key_jd].map(f_earth_y)
    df["z_earth"] = df[key_jd].map(f_earth_z)
    return df


def format4inv_query(df, key_jd, target, code="500", chunk_size=50):
    """Format lightcurves for convex inversion.

    Parameters
    ----------
    df : str
        dataframe of lightcurve 
    key_jd : str
        keyword of time in JD
    target : str
        target of interest
    code : str
        IAU observatory code
    chunk_size : int
        number of data to be queried at the same time

    Return
    ------
    df_format : pandas.DataFrame
        formatted data
    """

    # For observatory not registered ==========================================
    if code == "Seimei":
        # Slightly different from 371
        code = {
            "lon":        133.5967,
            "lat":         34.5769,
            "elevation" :   0.355
            }
    else:
        pass
    # For observatory not registered ==========================================

    jd_list = df[key_jd].tolist()
    x_S_all, y_S_all, z_S_all = [], [], []
    x_E_all, y_E_all, z_E_all = [], [], []

    for i in range(0, len(jd_list), chunk_size):
        jd_chunk = jd_list[i:i+chunk_size]

        vec_sun = Horizons(
            location="500@10", id=target, epochs=jd_chunk).vectors(refplane="ecliptic")
        x_S_all.extend(-vec_sun["x"].data)
        y_S_all.extend(-vec_sun["y"].data)
        z_S_all.extend(-vec_sun["z"].data)

        vec_earth = Horizons(
            location=code, id=target, epochs=jd_chunk).vectors(refplane="ecliptic")
        x_E_all.extend(-vec_earth["x"].data)
        y_E_all.extend(-vec_earth["y"].data)
        z_E_all.extend(-vec_earth["z"].data)

    df["x_sun"]   = x_S_all
    df["y_sun"]   = y_S_all
    df["z_sun"]   = z_S_all
    df["x_earth"] = x_E_all
    df["y_earth"] = y_E_all
    df["z_earth"] = z_E_all

    ## 1. Query vectors with JD and mpc code
    #for idx_r, row in df.iterrows():
    #    jd = row[key_jd]

    #    # Method 1. ===========================================================
    #    # Use vectors
    #    S = Horizons(location="500@10", id=target, epochs=jd)
    #    vec = S.vectors(refplane="ecliptic")
    #    # Vector from the Sun to asteroid (Sun -> ast)
    #    x_S, y_S, z_S = vec["x"][0], vec["y"][0], vec["z"][0]
    #    # Vector from asteroid to the Sun (ast -> Sun)
    #    x_sun, y_sun, z_sun = -x_S, -y_S, -z_S

    #    # Location of code means observer center
    #    E = Horizons(location=code, id=target, epochs=jd)
    #    vec = E.vectors(refplane="ecliptic")
    #    eph = E.ephemerides()

    #    # Vector from the Earth to asteroid (Earth -> ast)
    #    x_E, y_E, z_E = vec["x"][0], vec["y"][0], vec["z"][0]
    #    # Vector from asteroid to the Earth (ast -> Earth)
    #    x_earth, y_earth, z_earth = -x_E, -y_E, -z_E
    #    # Method 1. ===========================================================

    #    # Method 2. ===========================================================
    #    # Query (lam, beta, r, ra, dec, delta) with JD and mpc code, then convert them to vectors.
    #    # The results are identical to the results using the old "format4conv" function,
    #    # while those are a bit diffent from Method 1.
    #    # Also the method 1 seems smarter in terms of programming.
    #    # ast = Horizons(location=code, id=target, epochs=jd)
    #    # eph = ast.ephemerides()
    #    # lam, beta, r   = eph["EclLon"], eph["EclLat"], eph["r"]
    #    # ra, dec, delta = eph["RA"], eph["DEC"], eph["delta"]

    #    # ## use cartisian method
    #    # c_eclip = SkyCoord(
    #    #     lam, beta, r, frame="heliocentricmeanecliptic", 
    #    #     unit=(u.deg, u.deg, u.au)
    #    #     )
    #    # c_eclip = c_eclip.cartesian

    #    # x_sun = -c_eclip.x.value[0]
    #    # y_sun = -c_eclip.y.value[0]
    #    # z_sun = -c_eclip.z.value[0]

    #    # # not icrs (), but gcrs(Geocentric Celestial Reference System)
    #    # c_radec = SkyCoord(
    #    #     ra=ra, dec=dec, distance=delta, frame="gcrs", unit=(u.hourangle, u.deg, u.au),
    #    # )
    #    # c_radec_car = c_radec.cartesian
    #    # #print("radec cartesian")
    #    # #print(f"{c_radec_car.x}, {c_radec_car.y}, {c_radec_car.z}")

    #    # c_g_eclip_mean = c_radec.transform_to(
    #    #     astropy.coordinates.builtin_frames.GeocentricMeanEcliptic)
    #    # c_g_eclip_mean = c_g_eclip_mean.cartesian
    #    # x_earth = -c_g_eclip_mean.x.value[0]
    #    # y_earth = -c_g_eclip_mean.y.value[0]
    #    # z_earth = -c_g_eclip_mean.z.value[0]
    #    # Method 2. ===========================================================

    #    x_S_list.append(x_sun)
    #    y_S_list.append(y_sun)
    #    z_S_list.append(z_sun)
    #    x_E_list.append(x_earth)
    #    y_E_list.append(y_earth)
    #    z_E_list.append(z_earth)

    ## predict x, y, z of the Sun and the Earth 
    #df["x_sun"]   = x_S_list
    #df["y_sun"]   = y_S_list
    #df["z_sun"]   = z_S_list
    #df["x_earth"] = x_E_list
    #df["y_earth"] = y_E_list
    #df["z_earth"] = z_E_list
    return df


def save4inv(
    result, absflux, random, key_jd, key_flux, key_fluxerr, out):
    """
    Save formatted text for convex inversion.

    Parameters
    ----------
    result : array of pandas.DataFrame
        array of formatted DataFrame
    absflux : bool
        whether flux is absolute
    random : bool
        whether shuffle data using fluxerr
    key_jd : str
        keyword of time in JD
    key_flux : str
        keyword of flux
    key_fluxerr : str
        keyword of flux uncertainty
    out : str
        output filename
    tbin : str, optional
        widht of time bin

    Return
    ------
    df_format : pandas.DataFrame

    """
    N = len(result)
    with open(out, "w") as f:
        # Number of lightcurves
        f.write(f"{N}\n")
        for n in range(N):
            # n-th lightcurve
            df = result[n]

            if random:
                # Monte Carlo 
                flux_mc = np.random.normal(
                    df[key_flux], df[key_fluxerr], len(df))
                df["flux1"] = flux_mc
            else:
                df["flux1"] = df[key_flux]

            # Absolute flux 1 (calibrated)
            if absflux:
                f.write(f"{len(df)} 1\n")
            # Relative flux 0
            else:
                f.write(f"{len(df)} 0\n")

            for idx,row in df.iterrows():
                if absflux:
                    flux = row["flux1"]
                else:
                    flux = row["flux1"]/np.median(df["flux1"])
                jd = row[key_jd]
                f.write(
                    f"{jd} {flux} "
                    f"{row['x_sun']} {row['y_sun']} {row['z_sun']} "
                    f"{row['x_earth']} {row['y_earth']} {row['z_earth']} "
                    f"\n")


def do_conv(lam, beta, lc, lcdir=".", inpdir=".", outdir="."):
    """
    Do convex inversion.
    All results are saved in outdir.

    Parameters
    ----------
    lam : float
      longitude
    beta : float
      latitude
    lc : str
      lightcurve
    lcdir : str
      directory for lc, optional
    inpdir : str
      directory for input file, optional
    outdir : str
      directory for output file, optional
    """
    l = lam
    b = beta
    print(f"  (lam, beta) = ({l:.1f}, {b:.1f})")

    inp = f"input_ci_{int(l)}_{int(b)}"
    outarea = f"outarea_ci_{int(l)}_{int(b)}"
    outpar = f"outpar_ci_{int(l)}_{int(b)}"
    outlc = f"outlcs_ci_{int(l)}_{int(b)}"
    # Include chi2
    res = f"res_ci_{int(l)}_{int(b)}"
    cmd = (
        f"cat {lcdir}/{lc} | convexinv -v -o {outdir}/{outarea} "
        f"-p {outdir}/{outpar} {inpdir}/{inp} {outdir}/{outlc} > {outdir}/{res}"
        )
    subprocess.run(cmd, shell=True)


def do_conv_final(lc, inp="input.txt", outdir="."):
    """
    Do convex inversion.
    All results are saved in outdir.

    Parameters
    ----------
    lc : str
      lightcurve
    inpdir : str
      directory for input file, optional
    outdir : str
      directory for output file, optional
    """

    # Assume the filename is like inp_X_Y
    common_str = inp[4:]
    model = f"model_{common_str}"
    outlc = f"outlcs_{common_str}"
    outpara = f"outpara_{common_str}"

    cmd = (
        f"cat {lc} | convexinv -s -p {outdir}/{outpara} {inp} {outdir}/{outlc} "
        f"| minkowski | standardtri > {outdir}/{model}"
        )
    print(f"Execute\n  {cmd}")
    subprocess.run(cmd, shell=True)


def calc_JPLephem(asteroid, date0, date1, step, obscode, air=False):
    """

    Calculate asteroid ephemeris.
  
    Parameters
    ----------
    asteroid : str
      asteroid name like "Ceres", "2019 FA" (should have space)
    date0 : str
      ephemeris start date like "2020-12-12"
    date1 : str
      ephemeris end date like "2020-12-12"
    step : str
      ephemeris step date like '1d' for 1-day, '30m' for 30-minutes
    obscode : str, optional
      IAU observation code. 
      371:Okayama Astronomical Observatory
      381:Kiso observatory
    air : bool, optional
      whether consider air pressure
  
    Return
    ------
    ephem : astropy.table.table.Table
      calculated ephemeris
    """
  
    obj = Horizons(id=asteroid, location=obscode,
          epochs={'start':date0, 'stop':date1, 'step':step})
    eph = obj.ephemerides(refraction=air)
    return eph


def nobs_lc(lc):
    """

    Count the number of observations of a lightcurve.

    Parameter
    ---------
    lc : str
      lightcurve (text file)

    Return
    ------
    N_obs : int
      the number of observations
    """
    with open (lc, "r") as f:
        lines = f.readlines()
        lines = [x.replace("\n", "") for x in lines]
        N_all = len(lines)
        N_lc = lines[0]
        N_header = int(N_lc) + 1
        N_obs = N_all - N_header
        return N_obs


def tbinning(
    df, tbin, key_t="jd", unit_t="d", key_flux="flux", key_fluxerr="fluxerr"):
    """

    Count the number of observations of a lightcurve.
    The unit of time bin is always second!

    Parameter
    ---------
    df : pandas.DataFrame
        input dataframe
    tbin : float
        width of time bin in "second"
    key_t : float
        keyword for time
    unit_t : str
        day (d), hour (h), minumte (m), or second (s)
    key_flux : str
        keyword for flux
    key_fluxerr : str
        keyword for fluxerr

    Return
    ------
    df : pandas.DataFrame
        dataframe with binned data (t, flux, and fluxerr)
    """

    col = df.columns.tolist()
    keys = [key_t, key_flux, key_fluxerr]
    assert set(keys) <= set(col), "Check the input columns."

    if unit_t == "s":
        twidth = tbin
    elif unit_t == "m":
        twidth = tbin/60.
    elif unit_t == "h":
        twidth = tbin/3600.
    elif unit_t == "d":
        twidth = tbin/3600./24.
 
    # Observation arc in arbitrary unit
    arc = np.max(df[key_t]) - np.min(df[key_t])
    # Time zero point
    t0 = np.min(df[key_t])

    # (maximum) number of data points after binning
    N = np.int(np.ceil(arc/twidth))
    print(f"twidth={twidth}, N={N}")
    x, y, yerr = [], [], []
    for n in range(N):
        df_bin = df[(df[key_t] >= t0 + n*twidth) & (df[key_t] < t0 + (n+1)*twidth)]
        N_bin = len(df_bin)
        if N_bin == 0:
            continue

        # Average
        # TODO weighted average
        x.append(np.mean(df_bin[key_t]))
        y.append(np.mean(df_bin[key_flux]))
        yerr.append(adderr(df_bin[key_fluxerr])/N_bin)

    df_bin = pd.DataFrame({key_t: x, key_flux: y, key_fluxerr: yerr})
    print(f"(before binning) N_data={len(df)}, mean fluxerr={np.mean(df[key_fluxerr]):.2f}")
    print(f"(after binning ) N_data={len(df_bin)}, mean fluxerr={np.mean(df_bin[key_fluxerr]):.2f}")
    return df_bin


def golden_spiral_G10(N):
    """Generate evenly distributed points on a sphere using a symmetric golden spiral.

    Parameters
    ----------
    N : int
        Half the number of points minus one. 
        The total number of output points will be 2N + 1.

    Returns
    -------
    lon : ndarray of shape (2N + 1,)
        Longitudes of the points in degrees, in the range [0, 360].

    lat : ndarray of shape (2N + 1,)
        Latitudes of the points in degrees, in the range [-90, 90].


    Reference
    ---------
    González 2010, Math Geosci (2010) 42: 49–64.
    """

    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    i = np.arange(-N, N + 1)
    lat = np.arcsin(2 * i / (2 * N + 1)) * 180 / np.pi

    # fractional part of i * golden ratio, mapped to [0, 1)
    frac = np.mod(i * phi, 1.0)
    # map to [0, 360)
    lon = frac * 360  
    return lon, lat


def mean_angular_spacing(lon, lat):
    """Estimate the mean angular spacing between points on a sphere in degrees.

    Parameters
    ----------
    lon : ndarray
        Longitudes of points in degrees.
    lat : ndarray
        Latitudes of points in degrees.

    Returns
    -------
    mean_spacing : float
        Mean angular spacing (degrees) between nearest neighbor points on the sphere.
    """
    # Convert to Cartesian coordinates on the unit sphere
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    xyz = np.vstack([x, y, z]).T

    # Use KDTree to find nearest neighbors
    tree = cKDTree(xyz)
    # first one is itself, second is nearest neighbor
    dist, _ = tree.query(xyz, k=2)  

    # Convert chord distance to angle (arc length)
    angular_dist_rad = 2 * np.arcsin(dist[:, 1] / 2)
    angular_dist_deg = np.rad2deg(angular_dist_rad)

    return np.mean(angular_dist_deg)


def calc_CI_chi2(dof, sigma=3):
    """Calculate confidense interval (CI) of chi2 distribution.
    
    Parameters
    ----------
    dof : int, optional 
        degree of freedom of chi2 distribution (N-M)
    sigma : int
      confidence level

    Return
    ------
    CI : float
        n-sigma confidence interval in percentage
    """
    # For reduced chi2
    CI = sigma*(2/dof)**0.5
    return CI


def calc_confidence_chi2(p_list, chi2_list, dof, sigma):
    """Estimate rotation period and its error.

    Parameters
    ----------
    p_list : array-like
        list of rotation periods
    chi2_list : array-like
        corresponding chi2 values
    dof : int
        degrees of freedom
    sigma : float
        confidence level
    
    Returns
    -------
    P_cand : array-like
        rotation period with chi2 smaller than chi2_3sigma
    chi2_cand : float
        chi-squared values corresponding to P_cand
    chi2_nsigma : float
        chi-squared boundary with n-sigma confidence level
    """
    
    # Normalize chi2 with the best value
    chi2_min = np.min(chi2_list)
    chi2_norm_list = [x/chi2_min for x in chi2_list]

    # n-sigma like boundary
    # confidence level of 0.9973
    # Since chi2 min is normalized to the unity above,
    # this boundary corresponds to those of Polishook 2014, Icarus, 241, 79 
    # and Cambioni+2021, Nature.       
    chi2_min_norm = np.min(chi2_norm_list)
    chi2_nsigma = chi2_min_norm + sigma*(2/dof)**0.5

    # Search periods with chi2 values smaller than chi2_3sigma
    P_cand, chi2_cand = [], []
    for p, c in zip(p_list, chi2_norm_list):
        if c < chi2_nsigma:
            P_cand.append(p)
            chi2_cand.append(c)
    return P_cand, chi2_cand, chi2_nsigma
