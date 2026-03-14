#%%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple
from scipy.ndimage import gaussian_filter1d
from skimage import transform
from generate_line_profile import generate_line_profile
from secondary_straightening import process
# %%
# %%
# %%
win = '6300'
wlslice_wide = (629.5,630.5)
wlslice_narrow = (630,630.3)
gslice = slice(-4,4)
PLOT = True
#%%
#skyspec
datadir = Path('straightened_sky_data')
fnames = list(datadir.glob(f'*{win}*.nc'))
fnames.sort()
skyds = xr.open_mfdataset(fnames[4])
skyda = skyds.intensity.sel(gamma = gslice)
#%%
skyda.isel(tstamp=65).plot()
#%%
sky_lp, sky_nlp = generate_line_profile(skyda,win ,wlslice_wide, wlslice_narrow,sigma=1, normalize=True, save = False)
if PLOT:
    skyda.mean('tstamp').plot()
    sky_lp.plot(y = 'gamma', color = 'red', ls = '--', lw = 0.5)
# %%
#solspec
solfn = Path('straightened_sol_data')
sfnames = list(solfn.glob(f'*sol*{win}*.nc'))
sfnames.sort()
solds = xr.open_dataset(sfnames[0])
solds = solds.rename({'avg':'intensity'})
solda = solds.intensity.sel(gamma = gslice)
sol_lp, sol_nlp = generate_line_profile(solda, win, wlslice_wide, wlslice_narrow, sigma= 1, normalize=True, save=False)
if PLOT:
    solda.plot()
    sol_lp.plot(y = 'gamma', color = 'red', ls = '--', lw = 0.5)
# %%
if PLOT:
    #raw line profiles
    plt.figure()
    sol_lp.plot(y = 'gamma', color = 'orange', label = 'SolSpec')
    sky_lp.plot(y = 'gamma', color = 'blue', label = 'SkySpec')
    plt.title(f'Line Profiles at {win} A')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Gamma (deg)')
    plt.legend()

    #normalized line profiles
    plt.figure()
    plt.plot(sky_nlp.line_profile, sky_lp.gamma, color = 'blue', label = 'SkySpec')
    plt.plot(sol_nlp.line_profile, sol_lp.gamma, color = 'orange', label = 'SolSpec')
    plt.title(f'Normalized Line Profiles at {win} A')
    plt.xlabel('Deviation in Wavelength (nm)')
    plt.ylabel('Gamma (deg)')
    plt.legend()


# %%
#skyspec
test = process(skyds, sky_nlp, centralwl=None)
output = test.intensity.isel(tstamp = 65)
input = skyds.intensity.sel(gamma=gslice).isel(tstamp = 65)

wlslice = slice(wlslice_wide[0], wlslice_wide[1])
# (input-output).sel(wavelength=wlslice).plot()
PLOT = True
if PLOT:
    fig,ax = plt.subplots(1,2, figsize=(10,5))
    input.sel(wavelength= wlslice).plot(ax=ax[0])
    ax[0].set_title('Input')
    ax[0].axvline(int(win)/10, color='red', ls='--', lw=0.5)

    output.sel(wavelength= wlslice).plot(ax=ax[1])
    ax[1].set_title('Output')
    ax[1].axvline(int(win)/10, color='red', ls='--', lw=0.5)

    fig.suptitle(f'Secondary Straightening at {win} A SkySPEC')
sky = output

# %%
#solspec
test = process(solds, sol_nlp, centralwl=None)
output = test.intensity
input = solds.intensity.sel(gamma=gslice)

wlslice = slice(wlslice_wide[0], wlslice_wide[1])
# (input-output).sel(wavelength=wlslice).plot()
if PLOT:
    fig,ax = plt.subplots(1,2, figsize=(10,5))
    input.sel(wavelength= wlslice).plot(ax=ax[0])
    ax[0].set_title('Input')
    ax[0].axvline(int(win)/10, color='red', ls='--', lw=0.5)
    output.sel(wavelength= wlslice).plot(ax=ax[1])
    ax[1].set_title('Output')
    ax[1].axvline(int(win)/10, color='red', ls='--', lw=0.5)
    fig.suptitle(f'Secondary Straightening at {win} A SOLSPEC')

sol = output
# %%
sky-= sky.min()
sky/=( sky.max()- sky.min())

# %%
plt.figure()
(sol-sky).plot()
# %%
