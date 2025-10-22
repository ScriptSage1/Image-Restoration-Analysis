import cv2 as cv
import numpy as np

def apply_GaussianNoise(image, sigma=50):

  image_32 = image.astype(np.float32)

  noise = np.random.normal(0, sigma, image_32.shape)

  noisy_image = image_32 + noise 

  noisy_image = np.clip(noisy_image, 0, 255)

  noisy_image = noisy_image.astype(np.uint8)

  return noisy_image