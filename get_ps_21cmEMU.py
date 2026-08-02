import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import time


rcParams.update({'font.size': 14})

from py21cmemu import Emulator

### Get the redshift values, xHI values and redshift at which 1d power spectrum values are avilable

# Load the 1D summaries test set
# test_1d_path = '/home/leon/Documents/codes/21cmEMU/docs/tutorials/test_database.h5' 
test_1d_path = 'data/test_database.h5' 


# only need the 1D power spectrum and xHI values

with h5py.File(test_1d_path, 'r') as f:
    # print("Test set keys:", list(f.keys())) # test set keys

    # Input parameters (11 astrophysical params)
    test_params = np.array(f['input_params'])

    # 1D summaries
    test_xHI = np.array(f['xHI'])
    # test_Tb = np.array(f['Tb'])
    # test_Ts = np.array(f['Ts'])
    # test_tau = np.array(f['tau'])
    test_PS_1D = np.array(f['PS_1D_seeds']) # shape is (Nparams, Nseeds, Nz, Nk)
    PS_redshifts = np.array(f['PS_redshifts'])
    k = np.array(f["k"])
    # UVLFs: (N, 7 z-bins, 60 magnitudes)
    # test_LFs_raw = np.array(f['UVLFs'])

    # Axes
    redshifts = np.array(f['redshifts'])
    # M_UV_all = np.array(f['M_UV'])
    # LF_zs = np.array(f['UVLF_redshifts'])
    limits = np.array(f["limits"])
    ap = np.array(f["astro_param_keys"])

# Filter valid samples (no NaN inputs)
valid_mask = ~np.isnan(test_params.mean(axis=1))
# print(f"Valid samples: {valid_mask.sum()} / {len(test_params)}")

test_params = test_params[valid_mask]
test_xHI = test_xHI[valid_mask]
# test_Tb = test_Tb[valid_mask]
# test_Ts = test_Ts[valid_mask]
# test_tau = test_tau[valid_mask]
# test_LFs_raw = test_LFs_raw[valid_mask]
test_PS_1D = test_PS_1D[valid_mask]
# m = np.logical_and(M_UV_all < -10, M_UV_all > -20)
# Trim UVLFs to M_UV in [-20, -10]
# M_UV = M_UV_all[m]  # Crop to 30 magnitudes
# test_LFs = test_LFs_raw[:, :, m]  # (N, 7, 30)

# print("\nData shapes:")
# print(f"  Params: {test_params.shape}")
# print(f"  xHI: {test_xHI.shape}")
# # print(f"  Tb: {test_Tb.shape}")
# # print(f"  Ts: {test_Ts.shape}")
# # print(f"  tau: {test_tau.shape}")
# # print(f"  UVLFs: {test_LFs.shape}")
# print(f"  Redshifts: {redshifts.shape}")



def simu_1d_ps(theta, redshift_value):
    """
    Simulate the 1D power spectrum using the 21cmEMU emulator for given parameters and redshift.

    Parameters:
    theta (array): Array containing the astrophysical parameters.
    redshift_value (float): The redshift value at which to compute the power spectrum.

    Returns:
    tuple: A tuple containing the 1D power spectrum and the corresponding k values.
    """
    # Initialize the emulator
    emu = Emulator(emulator="mcg", emulate_2d_ps=False)
    
    # Predict outputs using the emulator
    _params, outputs, errors = emu.predict(theta)
    
    # Extract the 1D power spectrum and redshifts
    ps_1d = outputs["PS"].value
    PS_redshifts = np.array(outputs["PS_redshifts"].value)
    
    # Find the index of the specified redshift
    index_z = np.where(np.round(PS_redshifts, 2) == redshift_value)[0][0]
    
    # Get the 1D power spectrum at the specified redshift
    ps_1d_at_z = ps_1d[index_z]
    
    return ps_1d_at_z 