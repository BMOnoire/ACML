import cv2
import numpy as np
import config as cfg
import janitor as jn
import matplotlib.pyplot as plt


pickle_dataset_path = cfg.general["pickle_path"] / "dataset.pickle"
dataset_list = jn.pickle_load(pickle_dataset_path)
dataset_rgb, dataset_rgb_norm, dataset_grayscale, dataset_grayscale_norm = dataset_list
image = dataset_rgb[0][5]

img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

y0, u0, v0 = cv2.split(img_yuv)


# mapping u and v range
colormap_u = np.array([[[i, 255-i, 0] for i in range(256)]], dtype=np.uint8)
colormap_v = np.array([[[0, 255-i, i] for i in range(256)]], dtype=np.uint8)

# Convert back to BGR so we can apply the LUT and stack the images
y = cv2.cvtColor(y0, cv2.COLOR_GRAY2BGR)
u = cv2.cvtColor(u0, cv2.COLOR_GRAY2BGR)
v = cv2.cvtColor(v0, cv2.COLOR_GRAY2BGR)

u_mapped = cv2.LUT(u, colormap_u)
v_mapped = cv2.LUT(v, colormap_v)

yuv_recreation = cv2.merge((y0, u0, v0))
image_recreated = cv2.cvtColor(yuv_recreation, cv2.COLOR_YUV2BGR)


#cv2.imshow('normal', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.imshow('y', y)
#cv2.waitKey(0)
#cv2.imshow('u', u_mapped)
#cv2.waitKey(0)
#cv2.imshow('v', v_mapped)
#cv2.waitKey(0)
#cv2.resizeWindow("output", 400, 300)
result = np.vstack([img, y, u_mapped, v_mapped])
#cv2.imshow('merge', result)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#cv2.imwrite('shed_combo.png', result)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1600, 1600)
dataset = dataset_rgb[0]
result = np.vstack([np.hstack([dataset[0], dataset[0]])])

for i in range(1, 5):
    result = np.vstack([result, np.hstack([dataset[i], dataset[i]])])

cv2.imshow('image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
asd = 1