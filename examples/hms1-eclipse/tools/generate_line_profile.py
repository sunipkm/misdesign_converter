#%%
from __future__ import annotations

from typing import Iterable, Tuple
from tqdm import tqdm
import xarray as xr
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from uncertainties import ufloat, unumpy as unp
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

#%%
def generate_line_profile(da:xr.DataArray, 
                     win:str, wlslice_wide:Tuple[float, float], 
                     wlslice_narrow:Tuple[float, float], 
                     normalize:bool, 
                     sigma:float=5.0,
                     save:bool=False,
                     prefix:str='') -> Tuple[xr.DataArray, xr.Dataset]:
    """Generate line profile for secondary straightening.

    Args:
        da (xr.DataArray): intensity dataarray whose coordinates must include 'wavelength' and 'gamma'.
        win (str): processing window.
        wlslice_wide (Tuple[float, float]): wide region of interest. May be centered around the feature of interest. (min,max)
        wlslice_narrow (Tuple[float, float]): narrow region of interest. Should be within the wide region. (min,max)
        normalize (bool): whether to normalize the data.
        sigma (float): standard deviation for Gaussian kernel used in smoothing. Defaults to 5.0.   
        save (bool): whether to save the output. Defaults to False.
        prefix (str): prefix to add to the saved filename. Defaults to ''.

    Returns:
        Tuple[xr.DataArray, xr.Dataset]: line position DataArray and line profile Dataset (this will be saved to be used as input for secondary straightening).
    """    
    if isinstance(wlslice_wide, Tuple) or isinstance(wlslice_wide, list):
        wlslice_wide = slice(wlslice_wide[0], wlslice_wide[1])
    if isinstance(wlslice_narrow, Tuple) or isinstance(wlslice_narrow, list):
        wlslice_narrow = slice(wlslice_narrow[0], wlslice_narrow[1])

    #if the dataset has time dimension, sum over it
    da = da.sel(wavelength = wlslice_wide)
    if 'tstamp' in da.dims:
        da = da.sum('tstamp')
    if normalize:
        da = da - np.abs(da[-1]- da[0])
        da = da / (da.max(skipna=True) - da.min(skipna=True))
        # da = da - np.abs(da[-1]- da[0])
        # da = da / (da.max(skipna=True))

    #find the the wavelength where the intensity is minimum in the narrow slice
    da_input = da.sel(wavelength = wlslice_narrow)
    lp = da_input.idxmin('wavelength', skipna=True)

    #create smoothed line profile
    norm = lp.copy()
    norm.values = gaussian_filter1d(lp.values, sigma=sigma)
    norm_fac = np.nanmin(norm)
    norm -= norm_fac
    norm = norm.assign_attrs({'normalized_to_wl': float(norm_fac)})

    #convert to dataset
    norm = norm.to_dataset(name='line_profile')
    norm['line_profile'].attrs['units'] = 'nm'
    norm['gamma'].attrs = da.gamma.attrs
    norm['line_profile'].attrs['description'] = f'Line profile for {win} Angstrom created using line using L1A data. This data can be used for secondary straightening.'
    norm['DateCreated'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
    
    #save
    if save:
        outdir = Path('line_profiles')
        outdir.mkdir(exist_ok=True)
        norm.to_netcdf(outdir / f'{prefix}_line_profile_{win}.nc')

    return lp, norm

def main(skydatadir:Path,soldatadir:Path):
    bounds = pd.read_csv('bounds.csv', comment='#')
    bounds.set_index('id', inplace=True)
    bounds.index = bounds.index.astype(str)

    for win in tqdm(bounds.index):
        wlslice_wide = (bounds.loc[win,'widemin'], bounds.loc[win,'widemax'])
        wlslice_narrow = (bounds.loc[win,'narrowmin'], bounds.loc[win,'narrowmax'])
        sigma = bounds.loc[win,'sigma']
        gslice = slice(-4,4)

        #sky spectra
        fnames = list(skydatadir.glob(f'*{win}*.nc'))
        fnames.sort()
        skyds = xr.open_dataset(fnames[-1])
        skyda = skyds.intensity.sel(gamma = gslice)
        sky_lp, sky_nlp = generate_line_profile(skyda, win, wlslice_wide, wlslice_narrow, normalize=True, save=True, prefix='sky', sigma=sigma)
        
        #solspec
        sfnames = list(soldatadir.glob(f'*solspec*{win}*.nc'))
        sfnames.sort()
        solds = xr.open_dataset(sfnames[0])
        solds = solds.rename({'avg':'intensity'})
        solda = solds.intensity.sel(gamma = gslice)
        sol_lp, sol_nlp = generate_line_profile(solda, win, wlslice_wide, wlslice_narrow, normalize=True, save=True, prefix='sol', sigma=sigma)

def test(win:str):
    datadir = Path('line_profiles')
    sky_nlp = xr.open_dataset(list(datadir.glob(f'*sky*{win}*.nc'))[0])
    sol_nlp = xr.open_dataset(list(datadir.glob(f'*sol*{win}*.nc'))[0])
    fig, ax = plt.subplots(figsize=(8,5))
    sky_nlp.line_profile.plot(y='gamma', ax=ax, label='sky', color='blue')
    sol_nlp.line_profile.plot(y='gamma', ax=ax, label='sol', color='orange')
    ax.set_title(f'Line Profile for {win} A')
    ax.legend()
# %%
if __name__ == '__main__':
    skydatadir:Path=Path('straightened_sky_data')
    soldatadir:Path=Path('straightened_sol_data')
    main(skydatadir=skydatadir, soldatadir=soldatadir)
    # test()

# %%
