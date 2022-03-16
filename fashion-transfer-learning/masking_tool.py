import matplotlib.pyplot as plt

import numpy as np
from skimage import draw
from matplotlib import path


def get_mask_from_polygon_mpl(image_shape, polygon):
  """Get a mask image of pixels inside the polygon.

  Args:
    image_shape: tuple of size 2.
    polygon: Numpy array of dimension 2 (2xN).
  """
  xx, yy = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
  xx, yy = xx.flatten(), yy.flatten()
  indices = np.vstack((xx, yy)).T
  mask = path.Path(polygon).contains_points(indices)
  mask = mask.reshape(image_shape)
  mask = mask.astype('bool')
  return mask


def get_mask_from_polygon_skimage(image_shape, polygon):
  """Get a mask image of pixels inside the polygon.

  Args:
    image_shape: tuple of size 2.
    polygon: Numpy array of dimension 2 (2xN).
  """
  vertex_row_coords = polygon[:, 1]
  vertex_col_coords = polygon[:, 0]
  fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, image_shape)
  mask = np.zeros(image_shape, dtype=bool)
  mask[fill_row_coords, fill_col_coords] = True
  return mask




image_shape = (2000, 2000)
polygon = np.array([[80, 111, 146, 234, 407, 300, 187, 45], [465, 438, 499, 380, 450, 287, 210, 167]]).T
#polygon = np.array([[200, 300, 300, 200], [400, 390, 383, 394]]).T

# matplotlib
mask1 = get_mask_from_polygon_mpl(image_shape, polygon).astype('uint')
plt.imshow(mask1)
plt.show()
# scikit-image
mask2 = get_mask_from_polygon_skimage(image_shape, polygon).astype('uint')
plt.imshow(mask2)
plt.show()

plt.imshow(mask1 - mask2)
plt.colorbar()
plt.show()