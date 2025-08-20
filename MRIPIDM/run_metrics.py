#loaded model without having to resample


############################Try running again, should have less error than, error of 50 presents the error in the last phase line only

from sample import *

import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 2, figsize=(10, 10))

for i, Mhat in enumerate(sampled_images.detach().cpu()):
  smallest_difference = 100
  for j, M in enumerate(data):
    difference = mse(M, Mhat)
    smallest_difference = min(smallest_difference, difference)
  print(f"image_{i} smallest_difference: {smallest_difference}")


  ax[0, 0].imshow(M[0, :, :], cmap="gray")
  ax[0, 0].set_title("Mx")
  ax[0, 0].axis("off")

  ax[1, 0].imshow(M[1, :, :], cmap="gray")
  ax[1, 0].set_title("My")
  ax[1, 0].axis("off")

  ax[2, 0].imshow(M[2, :, :], cmap="gray")
  ax[2, 0].set_title("Mz")
  ax[2, 0].axis("off")

  ax[0, 1].imshow(Mhat[0, :, :], cmap="gray")
  ax[0, 1].set_title("Mx")
  ax[0, 1].axis("off")

  ax[1, 1].imshow(Mhat[1, :, :], cmap="gray")
  ax[1, 1].set_title("My")
  ax[1, 1].axis("off")

  ax[2, 1].imshow(Mhat[2, :, :], cmap="gray")
  ax[2, 1].set_title("Mz")
  ax[2, 1].axis("off")


  plt.show()


height = Mhat.shape[1]
width = Mhat.shape[2]

print(f"Mhat_center_X: {Mhat[0, height//10, width//10]}")
print(f"M_center_X: {M[0, height//10, width//10]}")
print(f"Mhat_center_Y: {Mhat[1, height//10, width//10]}")
print(f"M_center_Y: {M[1, height//10, width//10]}")
print(f"Mhat_center_Z: {Mhat[2, height//10, width//10]}")
print(f"M_center_Z: {M[2, height//10, width//10]}")





