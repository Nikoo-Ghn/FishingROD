import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
from lmfit.models import GaussianModel, ConstantModel
from sklearn.linear_model import RANSACRegressor, LinearRegression


# =============================================================================
# CONFIGURATION
# =============================================================================


def config():
    """
    Return a dictionary containing all configuration parameters.
    All display or background subtraction bounds are specified in q-space,
    so that raw pixel numbers do not appear in the configuration.
    """
    cfg = {}
    
    cfg['FileDir']= '/data/lmcat/Beamtime_2024/Beamtime_2024_11_19_ID10_IH-MA-566/RAW_DATA/GronhBN_2_xr/GronhBN_2_xr_0001/'

    cfg['FileName'] = '/GronhBN_2_xr_0001.h5'
    cfg['sample_name'] = 'GronhBN_2_xr'

    cfg['ScanN']        = '16'
    
    cfg['peak_type'] = '10'  #which rod is it? 11 or 10 etc?

    
    # Acquisition parameters
    cfg['energy']       = 22.50      # keV
    cfg['SDD']          = 690        # mm
    cfg['PixS']         = 0.055      # mm
    
    # Display window in qxy (in 1/Å)

    cfg['wqxy']   = [2.87, 2.95]
    
    cfg['Icutoff']      = 90      # intensity cutoff
    cfg['bg_lim']       = 0.5       # fraction for background selection
    cfg['blur_sigma']    = 30

    # qz window for summing intensity (in 1/Å)

    cfg['wqz'] = [0.09, -0.03]
    
    # Expected rod location in qxy (in 1/Å)

    cfg['rod_qxy']    = [2.9, 2.92]
    
    cfg['hotpix_cutoff']= 99       # percentile for hot pixel detection
    
    # Background subtraction bounds (in 1/Å) for scattering removal

    cfg['bg2']    = [2.855, 2.88]

    # area to mask out
    # cfg['mask_range'] = {
    #                 'qxy': (2.905, 2.913),  # mask columns 50 to 100
    #                 'qz': (0.2, 0.35)    # mask rows 30 to 80
    #             }    
    # Gaussian fit bounds in qz (in reciprocal space)

    cfg['gaus_w'] = [0.09, 0]

    # final plot bounds (if you don't put anything it will simply center the peak in qxy and put the whole bound in qz)

    # cfg['xlim'] = [2.85,2.95]
    cfg['xlim'] = [2.87,2.93]
    # cfg['ylim'] = [-0.05, 0.35] 
    cfg['ylim'] = [-0.04, 0.35]
    cfg['vmin_vmax'] = [0.2, 6]
    
    return cfg
