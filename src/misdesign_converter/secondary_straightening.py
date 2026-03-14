# %%
from __future__ import annotations
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import numpy as np
import xarray as xr
from typing import Dict, Optional
from skimage import transform
# %%
LOGGER = logging.getLogger(__name__)


def process_ufunc(
    img: np.ndarray,
    line_profile: np.ndarray,
    wlppix: float,
    centralwl_offset: float
) -> np.ndarray:
    """ performs a skimage.tranform.warp() on an image to straighten the spectra.

    Args:
        img (np.ndarray): image of shape (gamma, wavelength)
        line_profile (np.ndarray): line profile of shape (gamma,)
        wlppix (float): wavelength per pixel (nm/pixel) 
        centralwl (float): central wavelength of the line profile (nm)

    Returns:
        np.ndarray: straightened image of same shape as input img
    """
    # create meshgrid of pixel indices for output image coordinates
    xpix = np.arange(img.shape[0])
    ypix = np.arange(img.shape[1])
    mxo, myo = np.meshgrid(xpix, ypix, indexing='ij')

    # create a modified y-coordinate map based on the line profile as the inverse of the distortion in the input image
    lp_pix = line_profile / wlppix
    mi = myo + lp_pix[np.newaxis, :].T  # straighten
    mi -= (centralwl_offset / wlppix)  # shift to central wavelength

    # create the inverse coordinate traform map (output coords -> input coords)
    imap = np.zeros((2, *(img.shape)))
    imap[0, :, :] = mxo  # output x-coords map remains the same
    imap[1, :, :] = mi  # modified output y-coords map

    straightened = transform.warp(img, imap, order=1, mode='edge', cval=np.nan)
    return straightened

# %%


def process(
        ds: xr.Dataset,
        lprof: xr.Dataset,
        centralwl: Optional[float] = None,
) -> xr.Dataset:
    """ performs secondary straightening on ds of dims (...,gamma,wavelength) using skiimage.transform.warp() in the core function.

    Args:
        ds (xr.Dataset): dataset containing the image with dims ( .., gamma, wavelength)
        lprof (xr.Dataset): dataset containing the line profile with dim (gamma,)

    Returns:
        xr.Dataset: dataset containing the straightened image with same dims as input ds
    """
    # select only the gamma range for which the line profile is defined
    ds = ds.sel(gamma=lprof.gamma)
    # nm per pixel to convert wavelength axis back to pixel units
    wlppix = float(np.mean(np.diff(ds.wavelength.data)))
    if centralwl is None:
        centralwl_offset = 0.0
    else:
        centralwl_offset = centralwl - \
            float(lprof.line_profile.attrs['normalized_to_wl'])
    needle = ('gamma', 'wavelength')
    ids = []
    for key in ds.data_vars.keys():
        dims = ds[key].dims
        if dims[-len(needle):] == needle:
            ids.append(key)
    if len(ids) == 0:
        raise ValueError(
            f'No data variable with dimensions {needle} found in dataset.')
    for id in ids:
        straightened_data = xr.apply_ufunc(
            process_ufunc,
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


class SecondaryStraighten:
    """Secondary straightening of images.
    """

    def __init__(
        self,
        line_prof_dir: Path,
    ):
        """Initialize a SecondaryStraighten object.

        Args:
            line_prof_dir (Path): Path for line profile maps. This directory must contain .nc files with line profiles, and a settings.json file describing the separator used to discern the kind and window of the line profiles, and the central wavelength setting for each window and kind. If CWL settings are not provided, they will default to 0.0, i.e. no shift applied to the line profile.

        Raises:
            ValueError: No line profiles found in the specified directory.
        """
        if line_prof_dir is None:
            line_prof_dir = (Path(__file__).parent / 'line_profiles')
        line_prof_dir = line_prof_dir.absolute().resolve()
        line_profs = list(line_prof_dir.glob('*.nc'))
        if len(line_profs) == 0:
            raise ValueError(
                f'No line profile files found in {line_prof_dir}.'
            )
        if not (line_prof_dir / 'settings.json').exists():
            raise ValueError(
                f'settings.json not found in {line_prof_dir}. Please provide a settings.json file with the necessary settings line profiles.')
        self._line_profiles: Dict[str, Dict[str, Path]] = {}
        isettings = json.loads((line_prof_dir / 'settings.json').read_text())
        sep: str = isettings['separator']
        settings: Dict[str, Dict[str, str | float]] = isettings['settings']
        for lp in line_profs:
            kind, window = lp.stem.split(sep)
            if kind not in self._line_profiles.keys():
                self._line_profiles[kind] = {}
            self._line_profiles[kind][window] = lp
        for kind, profs in self._line_profiles.items():
            if kind not in settings.keys():
                LOGGER.warning(
                    f'No settings found for line profile kind {kind} in settings.json. Defaulting to centralwl_offset = none (no shift applied to line profile) for all windows of this kind.')
                settings[kind] = {key: 'none' for key in profs.keys()}
            else:
                for window in profs.keys():
                    if window not in settings[kind].keys():
                        LOGGER.debug(
                            f'No setting found for line profile kind {kind} and window {window} in settings.json. Defaulting to centralwl_offset = none (no shift applied to line profile).')
                        settings[kind][window] = 'none'
        kinds = list(settings.keys())
        windows = set()
        for kind in kinds:
            for window in settings[kind].keys():
                windows.add(window)
        LOGGER.info(
            f'Secondary line profile Kinds: {', '.join(kinds)}; Windows: {', '.join(windows)}.'
        )
        self._settings = settings

    def windows(self, kind: str) -> set[str]:
        """Get the set of windows for which line profiles are available.

        Returns:
            set[str]: Set of windows for which line profiles are available.
        """
        return set(self._settings[kind].keys())

    def straighten(
            self,
            ds: xr.Dataset,
            kind: str,
            window: str,
    ) -> xr.Dataset:
        kind = kind.lower()
        window = window.lower()
        if kind not in self._line_profiles.keys():
            raise ValueError(
                f'Line profile kind {kind} not found. Available kinds: {list(self._line_profiles.keys())}')
        profs = self._line_profiles[kind]
        if window not in profs.keys():
            raise ValueError(
                f'Line profile for window {window} not found in kind {kind}. Available windows: {list(profs.keys())}'
            )
        cwl = self._settings[kind][window]
        if isinstance(cwl, str):
            if cwl == 'auto':
                central_wl = float(window)/10
            elif cwl == 'none':
                central_wl = None
            else:
                raise ValueError(
                    'Invalid string value for cwl. Should be "auto", "none", or a float value.')
        elif isinstance(cwl, (int, float)):
            central_wl = float(cwl)
        else:
            raise ValueError(
                'Invalid type for cwl. Should be a string ("auto", "none") or a float value.')
        with xr.open_dataset(profs[window]) as lprof:
            newds = process(ds, lprof, centralwl=central_wl)
            if 'description' in newds.attrs.keys():
                newds.attrs['description'] += f' - Secondary straightened using {kind} line profile for {window}'
            else:
                newds.attrs['description'] = f'Secondary straightened using {kind} line profile for {window}'
            newds.attrs['modified'] = datetime.now(
                timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
            return newds
