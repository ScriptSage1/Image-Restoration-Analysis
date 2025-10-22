import cv2 as cv

def apply_DownSampling(image,scale_factor=0.5):

  h, w = image.shape[:2]

  new_h , new_w = int(h*scale_factor), int (w*scale_factor)

  image_down = cv.resize(image,(new_w,new_h),interpolation = cv.INTER_AREA)

  image_up = cv.resize(image_down, (w, h), interpolation = cv.INTER_LINEAR)

  return image_up