#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot convex inversion results.
Uncertainty is estimated in the same manner as the previous papers
such as Vokrouhlicky et al. (2011, 2017) and Durech et al. (2018).
The lightcurve is needed as an argument to calculate the number of observations N.


TODO
----
To calculate the number of degree of freedom, nu = N - M,
this code assumes M = 100 (i.e., nu = N - 100).
This does not change the result for normal shape modeling (N >> 100),
but I personally feel the exact number (M and N) should be writen 
in the manuscript if you are writing a paper.


References
----------
Vokrouhlicky et al. 2011, AJ, 142, 159.
Vokrouhlicky et al. 2017, AJ, 153, 270.
Durech et al. 2018, AA, 609, A86.
"""
import os 
from argparse import ArgumentParser as ap
import numpy as np
import matplotlib.pyplot as plt  
import matplotlib
from scipy.interpolate import griddata # type: ignore

from smfl import nobs_lc, golden_spiral_G10, calc_CI_chi2


if __name__ == "__main__":
    parser = ap(description="Plot result of convex inversion.")
    parser.add_argument(
        "--Nlam", type=int, default=1, 
        help="number of ecliptic longitude")
    parser.add_argument(
        "--Nbeta", type=int, default=1, 
        help="number of ecliptic latitude")
    parser.add_argument(
        "--N_golden", type=int, default=None, 
        help="Number of poles")
    parser.add_argument(
        "--dof", type=int, default=1000, 
        help="degree of freedom ")
    parser.add_argument(
        "--sigma", type=float, default=3, 
        help="CI level")
    parser.add_argument(
        "--norm", action="store_true", default=False,
        help="Normalize chi2 by the minimum")
    parser.add_argument(
        "--cmap", type=str, default="coolwarm",
        help="Colormap")
    parser.add_argument(
        "--outtype", type=str, default="png",
        help="format of output figures")
    parser.add_argument(
        "--resdir", type=str, default="convex_result",
        help="Directory for results of convexinv")
    parser.add_argument(
        "--out", type=str, default="polemap.png",
        help="Output finename")
    args = parser.parse_args()
 
    # Examples of color map
    ## "coolwarm", "inferno", "viridis", "cividis"
    cm = args.cmap
    # Normalization
    norm = args.norm

    # Number of data
    if args.N_golden:
        lam, beta = golden_spiral_G10(args.N_golden)
        # Make grid
        data = np.c_[lam, beta]
        xx, yy  = np.meshgrid(np.linspace(0, 360, 200), np.linspace(-90, 90, 100))
    else:
        lam = np.linspace(0, 360, args.Nlam)
        beta = np.linspace(-90, 90, args.Nbeta)
        # Make grid
        xx, yy = np.meshgrid(lam, beta)
        data = np.c_[xx.ravel(), yy.ravel()]

    # List for chi2 values
    chi2_list = []
    rotP_list = []
    dfac_list = []
    l_list, b_list = [], []
    for x in data:
        l, b = x[0], x[1]
        res = f"res_ci_{int(l)}_{int(b)}"
        res = os.path.join(args.resdir, res)
        with open(res, "r") as f:

            # Extract last 2 + 5 columns
            lines = f.readlines()[-7:]
            # === example ===
            # 50  chi2 4.710719  dev 0.055289  alambda 0.001000
            #
            # lambda, beta and period (hrs): 360.000000 -90.000000 0.014347
            # phase function parameters: 0.000000 0.100000 0.000000 
            # Lambert coefficient: 0.1
            # plus a dark facet with area 1.28%
            # === example ===

            # Extract chi2
            line = lines[0]
            line = line.split("chi2")[-1]
            line = line.split("dev")[0]
            chi2 = line.strip(" ")

            # Extract rotP
            line = lines[2]
            #print(line)
            line = line.split(" ")
            rotP = line[-1].strip("\n")
            # hour to sec
            rotP = float(rotP)*3600.

            # Extract dfac (dark facet area)
            line = lines[5]
            line = line.split(" ")
            dfac = line[-1].strip("%\n")

            chi2_list.append(float(chi2))
            rotP_list.append(float(rotP))
            dfac_list.append(float(dfac))
            l_list.append(float(l))
            b_list.append(float(b))

    # Extract minimum chi-squared and relavant parameters
    idx_min = np.argmin(chi2_list)
    chi2_min = chi2_list[idx_min]
    dfac_min = dfac_list[idx_min]
    l_min = l_list[idx_min]
    b_min = b_list[idx_min]
    print(
        f"Minimum chi-squared {chi2_min} w/\n"
        f"  (lam, beta) = ({l_min}, {b_min}), darkfacet {dfac_min:.2f}%\n"
        f"  (lam, beta) = ({int(l_min)}, {int(b_min)}), darkfacet {dfac_min:.2f}%")
    print("")
    info = (
        f"Minimum chi-squared {chi2_min} w/\n"
        f"  (lam, beta) = ({int(l_min)}, {int(b_min)}), darkfacet {dfac_min:.2f}%")

    # See Fatka+2025
    # Assume nu = N - M ~ N (i.e., N >> M)
    dof = args.dof
    CI = calc_CI_chi2(dof, sigma=args.sigma)
    print(f"{args.sigma}-sigma CI is {CI:.2f}")

    if norm:
        Z_chi2 = [x/chi2_min for x in chi2_list]
        Z_chi2 = np.array(Z_chi2)
        label_chi2 = r"$\chi^2$ (normalized with min)"
        levels_CI = [1 + CI]
        print(f"{args.sigma}-sigma range is 1 to (1 + CI)")
        print(f"  -> chi2: 1 to {1 + CI:.2f}")
    else:
        Z_chi2 = np.array(chi2_list)
        label_chi2 = r"$\chi^2$"
        # Where from?
        levels_CI = [chi2_min*(1 + CI)]
        print(f"{args.sigma}-sigma range is chi2_min to chi2_min(1 + CI)")
        print(f"  -> chi2: {chi2_min:.2f} to {chi2_min*(1 + CI):.2f}")

    Z_rotP = np.array(rotP_list)
    Z_dfac = np.array(dfac_list)
    
    if args.N_golden:
        Z_chi2 = griddata((lam, beta), Z_chi2, (xx, yy), method='linear')
        Z_rotP = griddata((lam, beta), Z_rotP, (xx, yy), method='linear')
        Z_dfac = griddata((lam, beta), Z_dfac, (xx, yy), method='linear')

    else:
        Z_chi2 = Z_chi2.reshape(xx.shape)
        Z_rotP = Z_rotP.reshape(xx.shape)
        Z_dfac = Z_dfac.reshape(xx.shape)

    fig = plt.figure(figsize=(8, 14))
    
    # mollweide does not work well......
    #ax1 = fig.add_axes([0.15, 0.72, 0.63, 0.25], projection="mollweide")
    ax1 = fig.add_axes((0.15, 0.72, 0.63, 0.25))
    cax1 = fig.add_axes((0.80, 0.72, 0.03, 0.25))
    # Add info.
    ax1.text(0.05, 0.90, info, size=10, transform=ax1.transAxes)

    #ax2 = fig.add_axes([0.15, 0.40, 0.63, 0.25], projection="mollweide")
    ax2 = fig.add_axes((0.15, 0.40, 0.63, 0.25))
    cax2 = fig.add_axes((0.80, 0.40, 0.03, 0.25))

    #ax3 = fig.add_axes([0.15, 0.08, 0.63, 0.25], projection="mollweide")
    ax3 = fig.add_axes((0.15, 0.08, 0.63, 0.25))
    cax3 = fig.add_axes((0.80, 0.08, 0.03, 0.25))

    y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False) # type: ignore
    cax2.yaxis.set_major_formatter(y_formatter)
    cax2.xaxis.set_major_formatter(y_formatter)
    
    ax1.set_xlabel(r"$\lambda$ [deg]")
    ax2.set_xlabel(r"$\lambda$ [deg]")
    ax3.set_xlabel(r"$\lambda$ [deg]")
    ax1.set_ylabel(r"$\beta$ [deg]")
    ax2.set_ylabel(r"$\beta$ [deg]")
    ax3.set_ylabel(r"$\beta$ [deg]")

    levels_cont = 100
    im = ax1.contourf(
        xx, yy, Z_chi2, zorder=-1, cmap=cm, levels=levels_cont)
    cbar = fig.colorbar(
        im,  cax=cax1, orientation='vertical', label=label_chi2)

    # Add CI line
    ax1.contour(
        xx, yy, Z_chi2, levels_CI, linestyles=["dashed"], colors="white")
        #xx, yy, Z_chi2, linestyles=["dashed"], colors="white")
    im = ax2.contourf(xx, yy, Z_rotP, zorder=-1, cmap=cm)
    cbar = fig.colorbar(
        im,  cax=cax2, orientation='vertical', label=r"$Period [s]$")
    im = ax3.contourf(xx, yy, Z_dfac, zorder=-1, cmap=cm)
    cbar = fig.colorbar(
        im,  cax=cax3, orientation='vertical', label="Dark facet area [%]")

    plt.savefig(args.out)
