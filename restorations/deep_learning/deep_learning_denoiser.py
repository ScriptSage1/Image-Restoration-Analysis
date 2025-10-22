import torch
import numpy as np
import os
import urllib.request
from torchvision.transforms.functional import to_tensor, to_pil_image

# Import the model architecture from the local blueprint file
from .drunet_model import DRUNet

# Global variable to store the loaded model to prevent reloading
g_model = None

def download_weights(url, model_dir="restorations/deep_learning/models"):
    """Downloads model weights if they don't exist."""
    os.makedirs(model_dir, exist_ok=True)
    filename = os.path.basename(url)
    model_path = os.path.join(model_dir, filename)
    if not os.path.exists(model_path):
        print(f"Downloading pre-trained model weights to {model_path}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(model_path, 'wb') as out_file:
                out_file.write(response.read())
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            return None
    return model_path

def apply_deep_learning_denoiser(image_rgb, sigma=25):
    """
    Applies the pre-trained DRUNet color model for Gaussian denoising.
    """
    global g_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("CUDA not found. Running on CPU, this may be slow.")

    # Load the model only once
    if g_model is None:
        print("Loading deep learning model (DRUNet Color)...")
        g_model = DRUNet(in_channels=4, out_channels=3)
        
        model_url = 'https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth'
        model_path = download_weights(url=model_url)
        if not model_path:
            print("!!! Failed to download model weights. Cannot denoise.")
            return image_rgb

        state_dict = torch.load(model_path, map_location=device)
        g_model.load_state_dict(state_dict)
        g_model.eval()
        g_model = g_model.to(device)
        print("Deep learning model loaded successfully.")
    
    # Create the 4-CHANNEL input (RGB + Noise Map)
    img_tensor = to_tensor(image_rgb).unsqueeze(0).to(device)
    noise_level = torch.FloatTensor([sigma / 255.0]).to(device)
    noise_level_map = torch.ones((1, 1, img_tensor.size(2), img_tensor.size(3)), device=device) * noise_level
    input_tensor = torch.cat((img_tensor, noise_level_map), 1)

    with torch.no_grad():
        denoised_tensor = g_model(input_tensor)
    
    denoised_numpy = denoised_tensor.squeeze(0).cpu().numpy()
    denoised_image = np.transpose(denoised_numpy, (1, 2, 0))
    denoised_image = np.clip(denoised_image, 0, 1) * 255
    denoised_rgb = denoised_image.astype(np.uint8)

    return denoised_rgb