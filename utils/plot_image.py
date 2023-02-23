import matplotlib.pyplot as plt
import numpy as np

def show_slices(image, layer, cmap):
    image = image.cpu()
    plt.figure(figsize=(10, 10))

    plt.subplot(131)
    plt.title("Sagittal")
    plt.xlabel(f"Layer: {layer[0]}")
    plt.imshow(np.rot90(image[layer[0], :, :]), cmap=cmap)

    plt.subplot(132)
    plt.title("Coronal")
    plt.xlabel(f"Layer: {layer[1]}")
    plt.imshow(np.rot90(image[:, layer[1], :]), cmap=cmap)

    plt.subplot(133)
    plt.title("Axial")
    plt.xlabel(f"Layer: {layer[2]}")
    plt.imshow(np.rot90(image[:, :, layer[2]]), cmap=cmap)

    plt.show()