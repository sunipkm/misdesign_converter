# %%
from misdesigner import MisInstrumentModel, MisCurveRemover
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from astropy.io import fits
from PIL import Image
# %%
model = MisInstrumentModel.load('hms_eclipse.json')
# %%
mmap = model.mosaic_map(unique=True, report=True)
predictor = MisCurveRemover(model, mmap)
imgsize = (len(predictor.beta_grid), len(predictor.gamma_grid))
# %%
data = fits.open('../hms1_solspec/20240507/20240507/hitmis_14ms_0_0_1715118896929.fit')[1].data.astype(float)
data = Image.fromarray(data)
data = data.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
image = Image.new('F', imgsize, color=np.nan)
image.paste(data, (45, -65))
data = np.asarray(image).copy()
data = xr.DataArray(
    data,
    dims=['gamma', 'beta'],
    coords={'gamma': predictor.gamma_grid,
            'beta': predictor.beta_grid},
)
fig, ax = model.plot_lines(figsize=(8, 6), dpi=300)
data.plot(ax=ax)
plt.show()

# %%
for win in predictor.windows:
    ds = predictor.straighten_image(data, win, coord='Slit')
    ds = ds.assign_coords(gamma=ds.gamma + 3.34)
    if np.all(ds.gamma > 0):
        ds = ds.assign_coords(gamma=-np.rad2deg(np.arctan((ds.gamma - 12.5) / 75)))
    else:
        ds: xr.DataArray = ds.loc[dict(gamma=slice(None, 0))]
        ds = ds.assign_coords(gamma=-np.rad2deg(np.arctan((ds.gamma + 12.5) / 75)))
    ds = ds.sortby('gamma')
    ds.plot()
    plt.show()
# %%
