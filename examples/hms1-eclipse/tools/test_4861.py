#%%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple
from scipy.ndimage import gaussian_filter1d
from skimage import transform
# %%
# %%
def get_line_profile(da:xr.DataArray, wlslice_wide:Tuple[float, float], wlslice_narrow:Tuple[float, float], normalize:bool) -> Tuple[xr.DataArray, xr.Dataset]:
    if isinstance(wlslice_wide, Tuple) or isinstance(wlslice_wide, list):
        wlslice_wide = slice(wlslice_wide[0], wlslice_wide[1])
    if isinstance(wlslice_narrow, Tuple) or isinstance(wlslice_narrow, list):
        wlslice_narrow = slice(wlslice_narrow[0], wlslice_narrow[1])

    da = da.sel(wavelength = wlslice_wide)
    if 'tstamp' in da.dims:
        da = da.sum('tstamp')
    if normalize:
        da = da - np.abs(da[-1]- da[0])
        da = da / (da.max(skipna=True) - da.min(skipna=True))
        # da = da - np.abs(da[-1]- da[0])
        # da = da / (da.max(skipna=True))
    
    da_input = da.sel(wavelength = wlslice_narrow)
    lp = da_input.idxmin('wavelength', skipna=True)

    norm = lp.copy()
    norm.values = gaussian_filter1d(lp.values, sigma=5)
    norm_fac = np.min(norm)
    norm -= norm_fac
    norm = norm.assign_attrs({'normalized_to_wl': float(norm_fac)})

    norm = norm.to_dataset(name='line_profile')
    norm['line_profile'].attrs['units'] = 'nm'
    norm['gamma'].attrs = da.gamma.attrs

    return lp, norm


# %%
win = '4861'
wlslice_wide = (485.5,486.8)
wlslice_narrow = (485.9,486.2)
gslice = slice(-4,4)
PLOT = False

#skyspec
datadir = Path('straightened_data')
fnames = list(datadir.glob(f'*{win}*.nc'))
fnames.sort()
skyds = xr.open_mfdataset(fnames[-1])
skyda = skyds.intensity.sel(gamma = gslice)
sky_lp, sky_nlp = get_line_profile(skyda, wlslice_wide, wlslice_narrow, normalize=False)
if PLOT:
    skyda.sum('tstamp').plot()
    sky_lp.plot(y = 'gamma', color = 'red', ls = '--', lw = 0.5)
# %%
#solspec
solfn = Path('solspec_data')
sfnames = list(solfn.glob(f'*solspec*{win}*.nc'))
sfnames.sort()
solds = xr.open_dataset(sfnames[0])
solds = solds.rename({'avg':'intensity'})
solda = solds.intensity.sel(gamma = gslice)
sol_lp, sol_nlp = get_line_profile(solda, wlslice_wide, wlslice_narrow, normalize=False)
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
    plt.plot(sky_nlp, sky_lp.gamma, color = 'blue', label = 'SkySpec')
    plt.plot(sol_nlp, sol_lp.gamma, color = 'orange', label = 'SolSpec')
    plt.title(f'Normalized Line Profiles at {win} A')
    plt.xlabel('Deviation in Wavelength (nm)')
    plt.ylabel('Gamma (deg)')
    plt.legend()

# %%
def secondary_straightening_core(img:np.ndarray, line_profile:np.ndarray, wlppix:float, centralwl_offset:float) -> np.ndarray:
    """ performs a skimage.tranform.warp() on an image to straighten the spectra.

    Args:
        img (np.ndarray): image of shape (za, wavelength)
        line_profile (np.ndarray): line profile of shape (za,)
        wlppix (float): wavelength per pixel (nm/pixel) 
        centralwl (float): central wavelength of the line profile (nm)

    Returns:
        np.ndarray: straightened image of same shape as input img
    """    
    #create meshgrid of pixel indices for output image coordinates
    xpix = np.arange(img.shape[0])
    ypix = np.arange(img.shape[1])
    mxo, myo = np.meshgrid(xpix, ypix, indexing='ij')

    #create a modified y-coordinate map based on the line profile as the inverse of the distortion in the input image
    lp_pix = line_profile/ wlppix
    mi = myo + lp_pix[np.newaxis,:].T #straighten
    mi -= (centralwl_offset / wlppix) #shift to central wavelength

    #create the inverse coordinate traform map (output coords -> input coords)
    imap = np.zeros((2,*(img.shape)))
    imap[0,:,:] = mxo #output x-coords map remains the same
    imap[1,:,:] = mi #modified output y-coords map

    straightened = transform.warp(img, imap, order=1, mode='edge', cval=np.nan)
    return straightened

#%%
def secondary_straightening(ds:xr.Dataset, lprof:xr.Dataset, centralwl:float) -> xr.Dataset:
    """ performs secondary straightening on ds of dims (...,za,wavelength) using skiimage.transform.warp() in the core function.

    Args:
        ds (xr.Dataset): dataset containing the image with dims ( .., za, wavelength)
        lprof (xr.Dataset): dataset containing the line profile with dim (za,)

    Returns:
        xr.Dataset: dataset containing the straightened image with same dims as input ds
    """    
    #select only the za range for which the line profile is defined
    ds = ds.sel(gamma = lprof.gamma)
    #nm per pixel to convert wavelength axis back to pixel units
    wlppix = float(np.mean(np.diff(ds.wavelength.data)))
    centralwl_offset = centralwl - float(lprof.line_profile.attrs['normalized_to_wl'])
    print(centralwl_offset)

    id = 'intensity'

    straightened_data = xr.apply_ufunc(
        secondary_straightening_core,
        ds[id],
        lprof.line_profile,
        input_core_dims=[['gamma', 'wavelength'], ['gamma']],
        output_core_dims=[['gamma', 'wavelength']],
        kwargs={'wlppix': wlppix, 'centralwl_offset': centralwl_offset},
        vectorize=True,
        dask='parallelized',
    )
    ds[id].data = straightened_data.data
    return ds
# %%
#skyspec
test = secondary_straightening(skyds, sky_nlp, centralwl=int(win)/10)
output = test.intensity.isel(tstamp = 0)
input = skyds.intensity.sel(gamma=gslice).isel(tstamp = 0)

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

    fig.suptitle(f'Secondary Straightening at {win} A SkySPEC')
sky = output

# %%
#solspec
test = secondary_straightening(solds, sol_nlp, centralwl=int(win)/10)
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
