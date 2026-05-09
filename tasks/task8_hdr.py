import cv2 as cv
import numpy as np


#firstly reading the images 

img1 = cv.imread('/content/under.jpeg')
img2 = cv.imread('/content/normal.jpeg')
img3 = cv.imread('/content/under.jpeg')
images = [img1, img2, img3]

#reading the exposure times through the camera phone image details 
times = np.array([1/17.0, 1/33.0, 1/50.0], dtype=np.float32)


#as they are multiple images so we need to make sure that every thing is aligned 
aligned = []
alignMTB = cv.createAlignMTB()
alignMTB.process(images, aligned)   # output goes into `aligned`
images = aligned

#CRF Camera response function 
calibrate = cv.createCalibrateDebevec()
response = calibrate.process(images, times)  # response shape: (256, 1, 3)


#Mergin into single HDR 
merge_debevec = cv.createMergeDebevec()
hdr = merge_debevec.process(images, times, response)  # float32 HDR
cv.imwrite('output.hdr', hdr)  # save raw HDR

#mapping again to make the result viewable using ReinHards' Algorithm + image clipping 
#tonemap = cv2.createTonemap(gamma=2.2)  # simplest
# or
tonemap = cv.createTonemapReinhard(gamma=1.5, intensity=0, light_adapt=0.8, color_adapt=0.0)
ldr = tonemap.process(hdr)
ldr_8bit = np.clip(ldr * 255, 0, 255).astype('uint8')
cv.imwrite('result.jpg', ldr_8bit)