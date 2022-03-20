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
polygon = np.array([[132, 154, 175, 182, 204, 216, 240, 248, 249, 268, 291, 307, 313, 315, 309, 315, 324, 336, 279, 220, 163, 141, 147, 146, 134, 131, 126, 135, 137, 132], [183, 179, 195, 204, 209, 211, 206, 198, 180, 185, 208, 230, 251, 255, 289, 324, 362, 425, 437, 446, 451, 430, 391, 361, 330, 309, 308, 254, 222, 181]]).T
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