import cv2 as cv

def apply_GaussianBlur(image_rgb,kernel_dim = 15,sigma=0):

  image_bgr = cv.cvtColor(image_rgb,cv.COLOR_RGB2BGR) # we need to convert rgb to bgr for opencv 

  blurred_bgr = cv.GaussianBlur(image_bgr,(kernel_dim,kernel_dim),sigma)

  blurred_rgb = cv.cvtColor(blurred_bgr,cv.COLOR_BGR2RGB)

  return blurred_rgb