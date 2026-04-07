import numpy as np
import h5py

import matplotlib
# matplotlib.use('Agg')  # non-GUI backend suitable for nodes , REMOVE if you want gui 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import animation
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
from lmfit.models import GaussianModel, ConstantModel
from sklearn.linear_model import RANSACRegressor, LinearRegression
import os
import pandas as pd


#for gif making
import imageio
from PIL import Image, ImageDraw, ImageFont

#to make simpler import of configs
import argparse
import importlib



# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def robust_baseline(qxy, intensity):
    X = qxy.reshape(-1, 1)
    ransac = RANSACRegressor(LinearRegression(), 
                                 residual_threshold=np.std(intensity)*0.5, 
                                 random_state=0)
    ransac.fit(X, intensity)
    baseline = ransac.predict(X)
    
    return baseline
def q_to_pixel(q_arr, q_val):
    """
    Given a monotonic array of q-values (for example, qxy as computed and flipped)
    and a target q value, return the pixel index whose q is closest to the target.
    """
    idx = np.argmin(np.abs(q_arr - q_val))
    return idx



def get_data(cfg):
    """
    Open the HDF5 file, retrieve the raw data and the direct beam positions,
    and compute the q-space arrays.
    Returns:
      f, Data0, PY0, PX0, qxy, qz, nxc, nyc,
      delta0, gam0, mu0  (the instrument positions in degrees).
    """
    f = h5py.File(cfg['FileDir'] + cfg['FileName'], "r")
    Data0 = f.get(cfg['ScanN'] + '.1/measurement/mpx_cdte_22_eh1')
    try:
        exposure_t = np.mean(f.get(cfg['ScanN'] + '.1/instrument/timer_delta/data'))    
    except:
        exposure_t = np.mean(f.get(cfg['ScanN'] + '.1/instrument/timer/data'))    

    Data0=Data0/exposure_t
    print(f"exposure = {exposure_t} s")


    try:
        PY0=cfg['PY0'] # along Qxy  , direct beam position at mu=0 gam=o delta=0
        PX0=cfg['PX0'] # along Qz  , direct beam position at mu=0 gam=o delta=0
    except:
        # Get direct beam positions from ROI2:
        PY0 = np.array(f.get(cfg['ScanN'] + '.1/instrument/mpx_cdte_22_eh1_roi2/selection/x')).item()
        PX0 = np.array(f.get(cfg['ScanN'] + '.1/instrument/mpx_cdte_22_eh1_roi2/selection/y')).item()
        
    # arm position [deg]
    delta0 = np.array(f.get(cfg['ScanN'] + '.1/instrument/positioners/delta')).item()
    gam0   = np.array(f.get(cfg['ScanN'] + '.1/instrument/positioners/gam')).item()
    mu0    = np.array(f.get(cfg['ScanN'] + '.1/instrument/positioners/mu')).item()

    Data0 = f.get(cfg['ScanN'] + '.1/measurement/mpx_cdte_22_eh1')
    th    = np.array(f.get(cfg['ScanN'] + '.1/instrument/th/value'))
    
    # Compute q-space arrays
    nic,nxc,nyc =np.shape(Data0)

    ResMap=np.ones((nic,nxc))
    qxy=np.array([0.0 + i for i in range(nxc)])
    rAngle= np.arctan((qxy-PY0)*cfg['PixS']/cfg['SDD'])   # relative angle, from th ereference pixel [rad]

    qz=np.array([0.0 + i for i in range(nyc)])
    r2Angle= np.arctan((PX0-qz)*cfg['PixS']/cfg['SDD'])   # relative angle, from th reference pixel [rad]

    # Calculate qxy and qz

    k0 = 2*np.pi/(12.398/cfg['energy'])

    phi = np.deg2rad(delta0)+rAngle
    psi = np.deg2rad(gam0) + r2Angle

    alpha_i = np.deg2rad(mu0)

    qx = k0*(np.cos(psi-alpha_i)*np.cos(phi) - np.cos(alpha_i))
    qy = k0*np.cos(psi-alpha_i)*np.sin(phi)
    qxy = np.sign(qy)*np.sqrt(qx**2 + qy**2)
    qxy = qxy[::-1] #because python's y axis 0 is on top for some reason...
    qz = k0*(np.sin(psi)+np.sin(alpha_i))
    
    
    return Data0, qxy, qz, th

def deadpix(Data0, cfg):
    """
    Computes a hot-pixel mask from dataset
    """
    nic = Data0.shape[0]
    stack = []
    diff_stack = []
    for i in range(nic):
        image=Data0[i, :, :]
        blurred = gaussian_filter(image, sigma=3)
        # Compute the difference image
        diff = image - blurred
        diff_stack.append(diff)
    
    diff_stack = np.array(diff_stack) 
    #median difference per pixel.
    median_diff = np.median(diff_stack, axis=0)

    # Determine the threshold from the specified percentile, e.g 99%
    threshold = np.percentile(median_diff, 98)
    deadpix_mask = median_diff > threshold
    deadpix_mask = np.flipud(deadpix_mask)
    plt.figure()
    plt.imshow(deadpix_mask, aspect='equal', cmap='turbo')

    return deadpix_mask


def sum_peaks(Data0, qxy, qz, th, cfg):
    """
    Process each scan to build summed images for the rod signal and for background,
    using a pre-determined hot pixel mask to zero out hot pixels before further processing.
    """

    # Pre-compute the hot pixel mask across the dataset.
    deadpix_mask= deadpix(Data0, cfg)
    
    # Convert the qz window from q-space to pixel indices.
    wqz_pix = [q_to_pixel(qz, cfg['wqz'][0]), q_to_pixel(qz, cfg['wqz'][1])]
    print("qz window (pixels):", wqz_pix)
    # Convert the qxy display window from cfg["wqxy"] to pixel indices. (Since qxy was flipped, the lower q value is at index 0.)
    wqxy_pix = [q_to_pixel(qxy, cfg['wqxy'][1]), q_to_pixel(qxy, cfg['wqxy'][0])]
    print("qxy window (pixels):", wqxy_pix)
    

    
    nic, nxc, nyc = np.shape(Data0)
    
    images_sum = np.zeros((nxc, nyc))
    images_bg  = np.zeros((nxc, nyc))
    hotpixels  = np.zeros((nxc, nyc))
    rod_qxy=cfg['rod_qxy']
    p = l = m = 0  # counts for rod images and background images respectively.

    peak_images = [] # list for storing individual peak images to make a gif
    th_value = []
    sum_fig= plt.figure()


    for i in range(nic):
        if i > 0:
            
            image = np.flipud(Data0[i, :, :])
            image=remove_scattering(image, qxy, cfg)

            
            image[deadpix_mask] = 0

            image=mask_rectangle(image,cfg,qxy,qz)
            image1 = image

            ImBlure = gaussian_filter(image1, sigma=cfg['blur_sigma'])   

            # Compute difference image (emphasizing peaks).
            gid = image - ImBlure
            
            # Sum intensity over the qz window.
            gid1 = np.sum(gid[:, wqz_pix[0]:wqz_pix[1]], axis=1)
            
            # Determine a local baseline from the intensity profile... it is not always linear, hence ransac

            baseline = robust_baseline(qxy[wqxy_pix[0]:wqxy_pix[1]], gid1[wqxy_pix[0]:wqxy_pix[1]])
            gid1[wqxy_pix[0]:wqxy_pix[1]] = gid1[wqxy_pix[0]:wqxy_pix[1]] - baseline

            plt.plot(qxy, gid1) #Plot all Is to see which we keep and cutoff

            Ipeak = np.max(gid1[wqxy_pix[0]:wqxy_pix[1]])
            Npos = np.argwhere(gid1 == Ipeak)[0]
            
            
            if 'im' in cfg and cfg['im'] is not None and i in cfg['im']:
                j = cfg['im'].index(i)
                images_sum += image
                p += 1
                
                # For the gif part:
                cropped_image = image[wqxy_pix[0]+40:wqxy_pix[1], :]
                peak_images.append(cropped_image.copy())
                
                th_i = th[i]
                th_value.append(th_i)
                print(i, Ipeak, qxy[Npos])
            
            #Automatic peak detection (if no manual selection exists)
            elif ('im' not in cfg or cfg['im'] is None) and Ipeak > cfg['Icutoff'] and (rod_qxy[0] <= qxy[Npos] <= rod_qxy[1]):
                images_sum += image
                p += 1
                
                # For the gif part:
                cropped_image = image[wqxy_pix[0]:wqxy_pix[1], :]
                peak_images.append(image.copy())
                th_i = th[i]
                th_value.append(th_i)
                print(i+1, Ipeak, qxy[Npos])
            
            # Background images
            elif Ipeak < cfg['Icutoff'] * cfg['bg_lim'] and not rod_qxy[0] < qxy[Npos] < rod_qxy[1]:
                images_bg += image
                l += 1
            
            # Fourth check: Hot pixels
            else:
                hotpix_threshold = np.nanpercentile(image, cfg['hotpix_cutoff'])
                hotpix_mask = image > hotpix_threshold
                hotpixels += image * hotpix_mask
                m += 1
            

    # Add reference lines for visualization.
    hline = plt.axhline(y=cfg['Icutoff'], color='blue', linestyle='--', linewidth=1)
    vline_low = plt.axvline(x=cfg['rod_qxy'][0], color='red', linestyle='--', linewidth=1)
    vline_high = plt.axvline(x=cfg['rod_qxy'][1], color='red', linestyle='--', linewidth=1)
    plt.legend([hline, vline_high], ['Icutoff', 'rod_qxy'])
    plt.xlim(cfg['wqxy'][0], cfg['wqxy'][1])
    plt.ylim(-1,cfg['Icutoff']*10)
    
    save_dir_1 = os.path.join("results", f"{cfg['sample_name']}_{cfg['ScanN']}")
    save_figure(sum_fig, cfg, suffix='I_cutoff', save_dir=save_dir_1, transparent=True)
    plt.show()
    print("Images with rod:", p, "Background images:", l)
    #terrible hot fix for when there is no backgound:
    if m==0:
        m=1
    if l==0:
        l=1
    map2D_peakonly = (images_sum / p)  - (hotpix_mask / m)  - (images_bg / l)
    map2D_peakonly[map2D_peakonly < 0] = 0



    fig1, axs = plt.subplots(2, 2, figsize=(12, 5))
    axs = axs.flatten()
    axs[0].imshow(images_sum / p, aspect='equal', extent=[qz[0], qz[-1], qxy[-1], qxy[0]], cmap='turbo', interpolation='nearest')
    axs[0].set_title('Sum of images with rod')
    axs[0].axhline(y=cfg['rod_qxy'][0], color='orange', linestyle='--', linewidth=2, alpha=0.8, label='rod_qxy')
    axs[0].axhline(y=cfg['rod_qxy'][1], color='orange', linestyle='--', linewidth=2, alpha=0.8)
    axs[0].axvline(x=cfg['wqz'][0], color='white', linestyle='--', linewidth=2, alpha=0.8, label='wqz')
    axs[0].axvline(x=cfg['wqz'][1], color='white', linestyle='--', linewidth=2, alpha=0.8)
    axs[0].set_ylim(cfg['wqxy'][0], cfg['wqxy'][1])
    axs[0].legend()
    
    axs[1].imshow(images_bg / l, aspect='equal', extent=[qz[0], qz[-1], qxy[-1], qxy[0]], cmap='turbo', interpolation='nearest')
    axs[1].set_title('Sum of background images')
    axs[1].set_ylim(cfg['wqxy'][0], cfg['wqxy'][1])
    
    axs[2].imshow(hotpixels / m, aspect='equal', extent=[qz[0], qz[-1], qxy[-1], qxy[0]], cmap='turbo', interpolation='nearest')
    axs[2].set_title('hot pixel mask')
    axs[2].set_ylim(cfg['wqxy'][0], cfg['wqxy'][1])


    axs[3].imshow(map2D_peakonly, aspect='equal', extent=[qz[0], qz[-1], qxy[-1], qxy[0]], cmap='turbo', interpolation='nearest')
    axs[3].set_title('Background removed (map2D_peakonly)')
    axs[3].set_ylim(cfg['wqxy'][0], cfg['wqxy'][1])
    axs[3].axhline(y=cfg['bg2'][0], color='red', linestyle='--', linewidth=2, alpha=0.8, label='bg2')
    axs[3].axhline(y=cfg['bg2'][1], color='red', linestyle='--', linewidth=2, alpha=0.8)
    axs[3].legend()
    
    plt.tight_layout()

    
    if peak_images:  #save each peak image as a gif, comment out if not needed
        #Compute overall normalization for the GIF frames.
        
        save_as_mp4(peak_images, th_value, cfg, extent=[qxy[-1], qxy[0],qz[-1], qz[0],])
        save_as_gif(peak_images, th_value, cfg, cmap='turbo',duration=0.75)
    
    return map2D_peakonly, p




def remove_scattering(map2D, qxy, cfg):
    """
    Remove scattering by subtracting a background line.
    The background is computed by averaging the rows in the window defined by
    cfg["bg2"] (given in q-space).
    Returns the background-subtracted 2D map.
    """
    bg_start = q_to_pixel(qxy, cfg['bg2'][1])
    bg_end   = q_to_pixel(qxy, cfg['bg2'][0])
    map2D_bkgr = map2D.copy()
    BKGRline = np.median(map2D[bg_start:bg_end, :], axis=0)
    for j in range(map2D.shape[0]):
        map2D_bkgr[j, :] = map2D[j, :] - BKGRline
    map2D_bkgr[map2D_bkgr < 0.1] = 0

    return map2D_bkgr


def save_as_gif(peak_images, th_value, cfg, cmap='turbo', duration=0.5):
    """
    Convert individual 2D numpy arrays (peak images) to RGB frames using the
    specified colormap and normalization, then save them as a GIF.
    
    Parameters:
        peak_images (list of np.array): List of 2D image arrays.
        gif_filename (str): Output GIF file name.
        cmap (str): The matplotlib colormap name.
        duration (float): Duration (in seconds) of each frame in the gif.
    """
    vmin_new = np.min(peak_images)
    vmax_new = np.max(peak_images)
    norm = plt.Normalize(vmin=vmin_new, vmax=vmax_new)
    colormap = plt.colormaps.get_cmap(cmap)
    frames = []
    scan_number = cfg['ScanN']
    base_name = cfg['sample_name']

    # Get normalization type from config; default is 'linear'
    norm_type = cfg.get('gif_norm', 'linear').lower()
    if norm_type == 'log':
        norm = colors.LogNorm(vmin=vmin_new, vmax=vmax_new)
    elif norm_type == 'symlog':
        # Use cfg['linthresh'] if provided, otherwise use a default (here 1% of the range)
        linthresh = cfg.get('linthresh', (vmax_new - vmin_new) * 0.01)
        norm = colors.SymLogNorm(linthresh=linthresh, vmin=vmin_new, vmax=vmax_new)
    else:
        norm = colors.Normalize(vmin=vmin_new, vmax=vmax_new)

        
    # Create results folder if it doesnt exist already
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)

    gif_filename = os.path.join(results_dir, f"{base_name}_scan{scan_number}_peaks.gif")

    for i, img in enumerate(peak_images):
        # Get an RGBA image using the colormap and then convert to an 8-bit RGB image.
        rgba = colormap(norm(img))
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        pil_img = Image.fromarray(rgb)

        #add number on image
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
        text = f"Th = {th_value[i]}"
        text_pos = (5, 5)
        draw.text(text_pos, text, font=font, fill=(255, 255, 255))

        frames.append(pil_img)

    imageio.mimsave(gif_filename, frames, duration=duration)
    print(f"Saved {len(frames)} peak images as gif: {gif_filename}")


def save_as_mp4(peak_images, th_value, cfg, extent=None):
    """
    similar to abov but mp4
    """
    # Global normalization across all frames
    
    vmin_new = float(np.min(peak_images))
    vmax_new = float(np.max(peak_images))
    
    norm = colors.Normalize(vmin=vmin_new, vmax=vmax_new)
    dpi=300
    fps=7
    cmap='turbo'

    # File paths
    scan_number = cfg.get('ScanN', '')
    base_name = cfg.get('sample_name', 'sample')
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    mp4_filename = os.path.join(results_dir, f"{base_name}_scan{scan_number}_peaks.mp4")

    peak_images = [np.rot90(img) for img in peak_images]
    peak_images = [np.fliplr(img) for img in peak_images]
    # Figure setup
    fig, ax = plt.subplots()
    im = ax.imshow(
        peak_images[0],
        cmap=cmap,
        norm=norm,
        origin='lower',
        extent=extent,
        interpolation='nearest',
        aspect='auto'   # <-- keep real aspect ratio
    )

    ax.set_xlabel(r'q$_{xy}$ (Å$^{-1}$)')
    ax.set_ylabel(r'q$_z$ (Å$^{-1}$)')
    ax.set_title(f"Sample {base_name}  scan #:{scan_number}")

    # On-figure text for th
    th_text = ax.text(
        0.02, 0.98, f"Theta = {th_value[0]}",
        transform=ax.transAxes, ha='left', va='top',
        color='w', fontsize=12, bbox=dict(facecolor='0.1', alpha=0.4, pad=3, edgecolor='none')
    )


    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Intensity (a.u)')

    # Update function
    def update(i):
        im.set_data(peak_images[i])
        th_text.set_text(f"Theta = {th_value[i]}")
        # Keep normalization fixed across frames
        return (im, th_text)

    # Animate and save
    anim = animation.FuncAnimation(fig, update, frames=len(peak_images), blit=False)
    writer = animation.FFMpegWriter(fps=fps, codec='libx264')
    anim.save(mp4_filename, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Saved {len(peak_images)} peak images as MP4: {mp4_filename}")

def gaussian_fit(map2D_bkgr, qz, qxy, cfg):
    """
    Sum the background-subtracted 2D map over a qz range (defined by cfg["gaus_w"])
    and fit the resulting qxy profile (within the window given by cfg["rod_qxy"])
    to a Gaussian plus constant model.
    Returns the fit result and the computed parameters.
    """
    # Get qz pixel indices for the gaussian fit window.
    wqz_gau = [q_to_pixel(qz, cfg['gaus_w'][0]),q_to_pixel(qz, cfg['gaus_w'][1])]
    
    flatten_line = np.mean(map2D_bkgr[:, wqz_gau[0]:wqz_gau[1]], axis=1) #mean foor normalization
    qxypix=np.arange(len(flatten_line))
    fit_min = qxypix[q_to_pixel(qxy, cfg['wqxy'][1])]
    fit_max = qxypix[q_to_pixel(qxy, cfg['wqxy'][0])]
    x_fit = qxypix[fit_min:fit_max]
    y_fit = flatten_line[fit_min:fit_max]

    # for when there is tilt
    baseline = robust_baseline(x_fit,y_fit)
    y_fit = y_fit - baseline
    y_fit[y_fit < 0] = 0

    plt.figure()
    plt.plot(x_fit, baseline, label="baseline")
    plt.plot(np.arange(len(flatten_line)), flatten_line, label="flattened")
    plt.plot(x_fit, y_fit+baseline, label="flattened")
    plt.show()

    gauss_model = GaussianModel(prefix='g_')
    const_model = ConstantModel(prefix='c_')
    model = gauss_model + const_model

    
    params = model.make_params()
    params['g_center'].set(value=np.mean(x_fit))
    params['g_amplitude'].set(value=(np.max(y_fit) - np.min(y_fit)))
    params['g_sigma'].set(value=(x_fit[-1] - x_fit[0]) / 4)
    params['c_c'].set(value=np.min(y_fit))
    
    result = model.fit(y_fit, params, x=x_fit)
    mean = result.params['g_center'].value
    sigma = result.params['g_sigma'].value

    #this value in pixels need to be deconvoluted from the measurement slit (1 pixel)
    sigma_deconv=np.sqrt((sigma)**2-1)
    fwhm= 2.35482 * sigma #absolute value
    fwhm_px = 2.35482 * sigma_deconv #decpnvoluted value

    pix = np.arange(len(qxy))

    # center in q
    mean_q = np.interp(mean, pix, qxy)

    # --- Sigma in q (map ±sigma_px around the center) ---
    q_left_sigma  = np.interp(mean - sigma_deconv, pix, qxy)
    q_right_sigma = np.interp(mean + sigma_deconv, pix, qxy)
    sigma_q = 0.5 * abs(q_right_sigma - q_left_sigma)

    # --- FWHM in q (map ±FWHM_px/2 around the center) ---
    fwhm_px = 2.35482 * sigma_deconv
    q_left_f  = np.interp(mean - 0.5 * fwhm_px, pix, qxy)
    q_right_f = np.interp(mean + 0.5 * fwhm_px, pix, qxy)
    fwhm_q = abs(q_right_f - q_left_f)

    print(f"Mean(q):  {mean_q:.6f}")
    print(f"Sigma(q): {sigma_q:.6e}")
    print(f"FWHM(q):  {fwhm_q:.6e}")

    print("Gaussian + Constant Fit Results:")
    print("Mean             : {:.6f}".format(mean))
    print("FWHM abs in px   : {:.6f}".format(fwhm))
    print("FWHM deconv in px: {:.6f}".format(fwhm_px))
    print("FWHM deconv in q : {:.6f}".format(fwhm_q))
    print("Baseline         : {:.6f}".format(result.params['c_c'].value))


    fig_fit, ax_fit = plt.subplots(figsize=(6, 4))
    ax_fit.plot(qxy, flatten_line, 'b.', label='Data')
    ax_fit.plot(qxy[fit_min:fit_max], result.best_fit + baseline, 'r-', label='Gaussian + Constant Fit')


    ax_fit.set_xlabel("qxy (1/Å)")
    ax_fit.set_ylabel("Intensity (a.u.)")

    plt.tight_layout()
    # ——— optional CSV dump ———
     
    save_dir= os.path.join("results", f"{cfg['sample_name']}_{cfg['ScanN']}")
    if save_dir is not None:
        
        os.makedirs(save_dir, exist_ok=True)
        df = pd.DataFrame({
            'qxy':   qxy[fit_min:fit_max],
            'data':  y_fit,
            'fit':   result.best_fit
        })
        csv_path = os.path.join(save_dir, f"{cfg['sample_name']}_{cfg['ScanN']}_gaussfit.csv")
        df.to_csv(csv_path, index=False)
        print(f"   ► saved fit data to {csv_path}")


    
    return result, mean_q, sigma_q, fwhm_q, x_fit, y_fit, flatten_line


def plot_map (qxy,qz,cfg,map2D): 
    fig2, ax2 = plt.subplots(figsize=(8, 2))
    ax2.set_title('Scan# '+str(cfg['ScanN'])+' , sum of '+str(p)+' images')
    ylim_idx=[q_to_pixel(qxy, cfg['rod_qxy'][1]),q_to_pixel(qxy, cfg['rod_qxy'][0])]
    subset = map2D[:,ylim_idx[0]:ylim_idx[1]]
    vmin_new = np.min(subset)
    vmax_new = np.max(subset)

    im_final=ax2.imshow(map2D, aspect='equal', extent=[qz[0],qz[-1], qxy[-1], qxy[0]],
        #   norm=colors.LogNorm(),cmap='turbo')
        #   vmin=0, vmax=vmax_new, cmap='turbo',interpolation='nearest')
          vmin=vmin_new, vmax=vmax_new, cmap='turbo',interpolation='nearest')
        # cmap='turbo',interpolation='nearest')
    # ax2.set_ylim(cfg['rod_qxy'][0], cfg['rod_qxy'][1])
    ax2.set_ylim(cfg['wqxy'][0], cfg['wqxy'][1])
    ax2.set_xlim(right=-0.05)

def plot_forpaper(qxy, qz, cfg, map2D, x_mean,results_dir=None):

    fig2, ax2 = plt.subplots(figsize=(4,15))
    plt.rcParams.update({'font.size': 24})
    # ax2.tick_params(axis='both', which='major', labelsize=14)

    # Get indices for cropping along qxy axis
    ylim_idx = [q_to_pixel(qxy, cfg['rod_qxy'][1]), q_to_pixel(qxy, cfg['rod_qxy'][0])]
    subset = map2D[:, ylim_idx[0]:ylim_idx[1]]


    # Rotate the image 90° clockwise
    rotated_map = np.rot90(map2D, k=-1)

    # === Handle axis limits ===
    # If cfg['xlim'] is defined, use it; otherwise, center around x_mean
    if 'xlim' in cfg:
        new_xlim = tuple(cfg['xlim'])
    else:
        new_xlim = (x_mean - 0.04, x_mean + 0.04)

    # If cfg['ylim'] is defined, use it; otherwise, use full qz range
    if 'ylim' in cfg:
        new_ylim = tuple(cfg['ylim'])
    else:
        new_ylim = (qz[-1], qz[0])
     
    # If cfg['vmin_vmax'] is defined, use it; otherwise, use the min and max intensity in window
    if 'vmin_vmax' in cfg:
        vmin_new = tuple(cfg['vmin_vmax'])[0]
        vmax_new = tuple(cfg['vmin_vmax'])[1]
    else:
        ylim_idx = [q_to_pixel(qxy, cfg['rod_qxy'][1]), q_to_pixel(qxy, cfg['rod_qxy'][0])]
        subset = map2D[:, ylim_idx[0]:ylim_idx[1]]
        vmin_new = np.min(subset)
        vmax_new = np.max(subset)
    
    # --- Set up the normalization object if specified in cfg ---
    norm_obj = None
    if 'plot_norm' in cfg:
        norm_type = cfg['plot_norm'].lower()
        if norm_type == 'log':
            norm_obj = colors.LogNorm(vmin=vmin_new, vmax=vmax_new)
        elif norm_type == 'symlog':
            # For symlog default to 1% of the range.
            linthresh = (vmax_new - vmin_new) * 0.01
            norm_obj = colors.SymLogNorm(linthresh=linthresh, vmin=vmin_new, vmax=vmax_new)
    else:
        # Default to linear 
        norm_obj = colors.Normalize(vmin=vmin_new, vmax=vmax_new)

    # Plot image
    
    im_final = ax2.imshow(rotated_map, aspect='equal',
                          extent=[qxy[-1], qxy[0], qz[-1], qz[0]],
                          norm=norm_obj,
                          cmap= 'turbo', interpolation='nearest')
    # cbar = fig2.colorbar(im_final, ax=ax2, norm=norm_obj)

    ax2.set_xlim(new_xlim)
    ax2.set_ylim(new_ylim)

    # Tick formatting
    ax2.tick_params(axis='both', which='major', labelsize=24)
    ax2.locator_params(axis='x', nbins=4)
    ax2.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    plt.xticks(rotation=45)
    ax2.locator_params(axis='y', nbins=6)
    # Axis labels
    # label_fontsize = 14
    # label_font = {'family': 'sans-serif', 'weight': 'bold', 'size': label_fontsize}
    ax2.set_xlabel(r'q$_{xy}$ (Å$^{-1}$)', fontsize=26)
    ax2.set_ylabel(r'q$_z$ (Å$^{-1}$)', fontsize=26)
    # plt.show()



    return fig2, ax2


def save_figure(fig, cfg, suffix=None, save_dir=None, transparent=True):
    """
    Save matplotlib figure
    """
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(save_dir, exist_ok=True)
    
    scan_number = cfg.get('ScanN')
    base_name = cfg.get('sample_name')
    plot_filename = os.path.join(save_dir, f"{base_name}_scan{scan_number}_{suffix}.png")
    
    fig.savefig(plot_filename, transparent=transparent)
    print(f"Saved figure as {plot_filename}")

def mask_rectangle(img,cfg,qxy,qz):
    """
    Applies a rectangular mask to the image based on the range defined in the configuration.
    The masked region will be set to 0.
    
    The configuration dictionary (cfg) should contain a key 'mask_range', which is expected
    to be a dictionary with:
       'x': a tuple (x_min, x_max)  -> column indices to mask,
       'y': a tuple (y_min, y_max)  -> row indices to mask.
    
    Parameters:
        image (np.ndarray): 2D array representing the image.
        cfg (dict): Configuration dictionary.
        
    Returns:
        np.ndarray: The image with the defined rectangular region set to 0.
    """
    if 'mask_range' not in cfg:
        return img  # No mask specified; return image as is

    # get mask boundaries from the config.
    mask_config = cfg['mask_range']
    # expect mask_config to have keys 'x' and 'y'
    qxy_range = mask_config.get('qxy')
    qz_range = mask_config.get('qz')
    
    # Convert the qz window from q-space to pixel indices.
    mqz_pix = [q_to_pixel(qz, qz_range[0]), q_to_pixel(qz, qz_range[1])]

    # Convert the qxy display window from cfg["wqxy"] to pixel indices. (Since qxy was flipped, the lower q value is at index 0.)
    mqxy_pix = [q_to_pixel(qxy, qxy_range[0]), q_to_pixel(qxy, qxy_range[1])]


    
    #avoid modifying the original.
    masked_image = img.copy()
    masked_image[mqxy_pix[1]:mqxy_pix[0], mqz_pix[1]:mqz_pix[0]] = 0
    
    return masked_image


def compute_peak_params(mean, sigma, fwhm, peak_type):
    """
    Compute lattice spacing (a), uncertainty (delta_a), and flake size (delta_d)
    for a given rodd type 11, 10 etc.

    Parameters:
        mean (float): Peak center in q-space
        sigma (float): Peak width (Gaussian sigma)
        fwhm (float): Full width at half maximum
        peak_type (str): Either '11' or '10'
    Returns:
        dict: {'q [1/Å]', 'a [Å]', 'size [µm]'}
    """
    delta_d = (2 * np.pi*0.9 / fwhm) / 1e4  # in µm
    if peak_type == '11':
        a_val = 4 * np.pi / mean
        delta_a_fit = (4 * np.pi / (mean ** 2)) * sigma #statistical error
        delta_a_foot = (4 * np.pi / (mean ** 2)) * 0.027  #footprint error
    elif peak_type == '10':
        a_val = 4 * np.pi / (np.sqrt(3) * mean)
        delta_a_fit = (4 * np.pi / (np.sqrt(3) * mean ** 2)) * sigma  #statistical error
        delta_a_foot = (4 * np.pi / (np.sqrt(3) * mean ** 2)) * 0.017 #footprint error
    else:
        raise ValueError(f"Unknown peak type: {peak_type}")

    return {
        'q [1/Å]': f"{mean:.5f} ± {sigma:.5f}",
        'a [Å]': f"{a_val:.5f}",
        'da_fit [Å]': f"{delta_a_fit:.5f}",
        'da_foot [Å]': f"{delta_a_foot:.5f}",
        'size [µm]': delta_d
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process measurement data using a specific config.")
    parser.add_argument("-c", "--config", default="myconfig",
                        help="Name of the config file (without .py) to use.")
    args = parser.parse_args()
   
    
    # dynamically import the config module from the config folder.
    config_module = importlib.import_module(f"config.{args.config}")
    cfg = config_module.config()
    

    Data0, qxy, qz, th= get_data(cfg)
    print("qxy shape:", qxy.shape, "qz shape:", qz.shape)
    
    map2D,p = sum_peaks(Data0, qxy, qz, th, cfg)

    # --- Remove scattering background ---
    map2D_bkgr = remove_scattering(map2D, qxy, cfg)
    
    # --- final map ---
    map2D=map2D_bkgr
    # --- Final 2D map plot ---



    plot_map (qxy,qz,cfg,map2D)
    
    result, mean, sigma, fwhm, x_fit, y_fit, flatten_line = gaussian_fit(map2D, qz, qxy, cfg)
    fig,ax=plot_forpaper(qxy, qz, cfg, map2D, mean) #final plot
    save_figure(fig, cfg, suffix='sum', save_dir=None, transparent=True)
    plt.show()
    plt.close()
    peak_type = cfg.get('peak_type')  # 11 or 10?
    computed_params = compute_peak_params(mean, sigma, fwhm, peak_type)

    for k, v in computed_params.items():
        print(f"{k}: {v}")

    