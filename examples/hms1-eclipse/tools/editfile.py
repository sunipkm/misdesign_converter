# %%
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Union
import numpy as np
import os
from astropy.io import fits
from glob import glob
from tqdm import tqdm
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import xarray as xr

# %%
def get_info(fname: str)->Tuple[float, float]:
    with fits.open(fname) as hdul:
        header = hdul[-1].header
        return header['TIMESTAMP']*1e-3, header['EXPOSURE_MS']*1e-3

# %%
files = glob('../../eclipse/data/*/*.fit*')
ifo = np.asarray(list(map(get_info, files))).T
sidx = np.argsort(ifo[0])
ifo = ifo[:, sidx]
# %%
idx = int(np.where(ifo[1] > 60)[0])
# %%
dtstamp = np.concatenate((np.diff(ifo[0, :idx-1]), [np.nan], np.diff(ifo[0, idx+1:])))
dexp = np.concatenate((ifo[1, 1:idx-1], [np.nan], ifo[1, idx+2:]))
res = dtstamp - dexp
plt.plot(res, ls='', marker='o')
# %%
a = np.diff(ifo[0, idx-10:idx])
b = ifo[1, idx-10:idx-1]
plt.plot(a, label='tdiff', ls='', marker='o')
plt.plot(b, label='exposure', ls='', marker='x')
plt.plot(a-b, label='residual', ls='', marker='+')
plt.plot(((a-b) < 0)*10, label='mask')
plt.legend()
# %%
a = np.diff(ifo[0, idx+1:idx+10])
b = ifo[1, idx+1:idx+9]
plt.plot(a, label='tdiff', ls='', marker='o')
plt.plot(b, label='exposure', ls='', marker='x')
plt.plot(a-b, label='residual', ls='', marker='+')
plt.plot(((a-b) < 0)*10, label='mask')
plt.legend()
# %%
a = np.diff(ifo[0, idx-10:idx])
b = ifo[1, idx-10:idx-1]
plt.plot(range(9), a, label='tdiff', ls='', marker='o', color='r')
plt.plot(range(9), b, label='exposure', ls='', marker='x', color='b')
plt.plot(range(9), a-b, label='residual', ls='', marker='+', color='k')
plt.plot(range(9), ((a-b) < 0)*10, label='mask', color='g')

a = np.diff(ifo[0, idx+1:idx+10])
b = ifo[1, idx+1:idx+9]
plt.plot(range(11, 19), a, ls='', marker='o', color='r')
plt.plot(range(11, 19), b, ls='', marker='x', color='b')
plt.plot(range(11, 19), a-b, ls='', marker='+', color='k')
plt.plot(range(11, 19), ((a-b) < 0)*10, color='g')
plt.legend()
# %%
a = np.diff(ifo[0, idx-10:idx])
b = ifo[1, idx-9:idx]
plt.plot(range(9), a, label='tdiff', ls='', marker='o', color='r')
plt.plot(range(9), b, label='exposure', ls='', marker='x', color='b')
plt.plot(range(9), a-b, label='residual', ls='', marker='+', color='k')
plt.plot(range(9), ((a-b) < 0)*10, label='mask', color='g')

a = np.diff(ifo[0, idx+1:idx+10])
b = ifo[1, idx+2:idx+10]
plt.plot(range(11, 19), a, ls='', marker='o', color='r')
plt.plot(range(11, 19), b, ls='', marker='x', color='b')
plt.plot(range(11, 19), a-b, ls='', marker='+', color='k')
plt.plot(range(11, 19), ((a-b) < 0)*10, color='g')

plt.plot(10, ifo[0, idx + 1]-ifo[0, idx], ls='', marker='o', color='k')
plt.legend()
# %%
fname = files[sidx[idx]]
with fits.open(fname) as hdul:
    header = hdul[-1].header
    header['EXPOSURE_MS'] = 30000
    os.rename(fname, fname.replace('.fits', '.bak'))
    fname = fname.replace('120000', '30000')
    hdul.writeto(fname)
# %%
