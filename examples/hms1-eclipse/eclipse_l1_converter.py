import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from astropy.io import fits
from PIL import Image
import numpy as np
from xarray import DataArray, Dataset

from misdesign_converter import L1Converter, MisCurveRemover, DetectorNoise

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)


def read(file: Path) -> Tuple[np.ndarray, float, float, float, Dict[str, Any]]:
    with fits.open(file) as hdul:
        if len(hdul) == 0:
            raise ValueError(
                f'FITS file {file} does not contain any HDUs.')
        elif len(hdul) == 1:
            hdu = hdul[0]
        elif len(hdul) == 2:
            hdu = hdul[1]
        else:
            raise ValueError(
                f'FITS file {file} contains more than 2 HDUs, which is unexpected.')
        data = np.array(hdu.data, dtype=np.float32)  # type: ignore
        header = dict(hdu.header)  # type: ignore
        tstamp = header['TIMESTAMP']*1e-3  # in ms, convert to s
        exposure = int(
            header['EXPOSURE_MS']
        )*1e-3  # in ms, convert to s
        temperature = header['CCDTEMP']

        return (data, tstamp, exposure, temperature, {})


def crop_and_resize(
    data: np.ndarray,
    predictor: MisCurveRemover
) -> DataArray:
    imgsize = (len(predictor.beta_grid), len(predictor.gamma_grid))
    idata = Image.fromarray(data)
    idata = idata.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    image = Image.new('F', imgsize, color=np.nan)
    image.paste(idata, (45, -65))
    idata = np.asarray(image).copy()
    # 5. Convert to DataArray
    image = DataArray(
        idata,
        dims=['gamma', 'beta'],
        coords={
            'gamma': predictor.gamma_grid,
            'beta': predictor.beta_grid,
        },
        attrs={'unit': 'ADU s^{-1}'},
    )
    return image


def finalize_flat(_: str, dataset: Dataset) -> Dataset:
    nds = dataset.assign_coords(
        gamma=dataset.gamma + 3.34
    )
    if np.all(nds.gamma > 0):
        nds = nds.assign_coords(
            gamma=-np.rad2deg(np.arctan((nds.gamma - 12.5) / 75))
        )
    else:
        nds: Dataset = nds.loc[dict(gamma=slice(None, 0))]
        nds = nds.assign_coords(
            gamma=-np.rad2deg(np.arctan((nds.gamma + 12.5) / 75))
        )
    nds = nds.sortby('gamma')
    nds.attrs.update(dict(
        description="HMS A - Straightened Eclipse data.",
    ))
    return nds


def getctime(fname: Path) -> float:
    if '_' not in fname.stem:
        raise ValueError(
            f'File name {fname} does not contain underscore to extract creation time.')
    words = fname.stem.split('_')
    return int(words[-1])*1e-3  # in ms, convert to s


def from_pickle(file: Path) -> Optional[DetectorNoise]:
    """Load detector noise from an LZMA compressed, `pickle`d file.

    Args:
        file (Path): LZMA compressed `pickle` file.

    Returns:
        DetectorNoise: Loaded detector noise.
    """
    import pickle
    import lzma
    with lzma.open(file, 'rb') as f:
        data = pickle.load(f)
    return DetectorNoise(dark=data['dark'], bias=data['bias'], readnoise=3.0)


converter = L1Converter.create(
    description='Convert HMS A Eclipse data from L0 to L1, with exposure normalization and Dark subtraction. It uses the MisInstrument model to extact ROI and performs line straightening. This program will not work without the instrument definition JSON file.',
    invocation_dir=Path(__file__).parent,
    timestamp=getctime,
    loader=read,
    ifilter=crop_and_resize,
    flatfinzalize=finalize_flat,
    detnoiseloader=from_pickle,
)
converter.process()
