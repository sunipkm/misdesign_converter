# %%
from __future__ import annotations
import logging
from pathlib import Path
import xarray as xr
from typing import Dict, List
# %%
LOGGER = logging.getLogger(__name__)


class FlatFieldCorrector:
    def __init__(self, path: Path):
        path = path.absolute().resolve()
        if not path.exists():
            raise FileNotFoundError(f'Flat field file {path} does not exist.')
        if path.is_file():
            raise ValueError(f'Flat field file {path} is not a directory.')
        flat_profs = list(path.glob('*.nc'))
        if len(flat_profs) == 0:
            raise ValueError(
                f'Flat field directory {path} does not contain any .nc files.')
        self._flat_fields: Dict[str, Path] = {}
        for fp in flat_profs:
            with xr.open_dataset(fp) as ds:
                self._flat_fields[ds.attrs['ROI']] = fp

    def apply(self, data: xr.Dataset, roi: str) -> xr.Dataset:
        if roi not in self._flat_fields:
            raise ValueError(f'No flat field found for ROI {roi}.')
        with xr.open_dataset(self._flat_fields[roi]) as ds:
            flat_field = ds['scale'].data
        for var in data.data_vars:
            if data[var].dims[-2:] != ('gamma', 'beta'):
                continue
            corrected = data[var].data / flat_field
            data[var].data = corrected
        return data

    @property
    def windows(self) -> List[str]:
        return list(self._flat_fields.keys())
