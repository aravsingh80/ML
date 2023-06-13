import cv2         
import numpy as np    
import matplotlib.pyplot as plt
from skimage import filters
from PIL import Image

image1 = cv2.imread('Catch.jpg')
image2 = Image.open('Catch.jpg')
  
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
  
ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + 
                                            cv2.THRESH_OTSU)     
image = thresh1
# cv2.imshow('Otsu Threshold', thresh1)  
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()   
edge_sobel = filters.sobel(image)

fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True,
                         figsize=(12, 6))

axes[0].imshow(image2)
axes[0].set_title('Original Image')

axes[1].imshow(thresh1, cmap=plt.cm.gray)
axes[1].set_title('Otsu')

axes[2].imshow(edge_sobel, cmap=plt.cm.gray)
axes[2].set_title('Sobel Edge Detection')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()