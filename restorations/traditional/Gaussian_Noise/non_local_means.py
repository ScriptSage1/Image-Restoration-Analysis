import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma

def apply_non_local_means(image_rgb):

    img_float = image_rgb.astype(np.float32) / 255.0

    sigma_est = np.mean(estimate_sigma(img_float, channel_axis=-1))

    denoised_float = denoise_nl_means(
        img_float,
        h=1.15 * sigma_est,  # Denoising strength parameter
        fast_mode=True,
        patch_size=5,        
        patch_distance=6,    
        channel_axis=-1
    )

    denoised_rgb = (denoised_float * 255).astype(np.uint8)

    return denoised_rgb