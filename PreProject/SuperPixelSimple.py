from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt

image = img_as_float(io.imread("C:/Users/Tibet/Desktop/icon3.jpg"))
segments = slic(image, n_segments=250, compactness=10, sigma=1 )
fig=plt.figure("superpixels-- %d segments"%(250))
ax = fig.add_subplot(1,1,1)
ax.imshow(mark_boundaries(image,segments,color=(1,1,1),outline_color=(1,1,1)))

plt.axis('off')
plt.show()

