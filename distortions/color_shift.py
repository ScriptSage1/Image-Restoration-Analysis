import numpy as np

def apply_ColorShift(image,del_r=0,del_g=0,del_b=0):

  shifted = image.astype(np.int16) #to avoid overflow and underflow 

  shifted[...,0] += del_r #shifting color for red channel

  shifted[...,1] += del_g #shifting color for blue channel 

  shifted[...,2] += del_b #shifting color the green channel

  shifted = np.clip(shifted,0,255)

  shifted = shifted.astype(np.uint8)

  return shifted