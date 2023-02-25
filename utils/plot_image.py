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


def show_labels(label, layer, nrows, ncols, index_all, label_dict):
    """ label: 5D tensor label"""
    plt.figure(figsize=(15, 15))
    for i in range(label.shape[1]):
        plt.subplot(nrows, ncols, i+1)
        plt.title(label_dict[index_all[i]])
        plt.imshow(np.rot90(label[0, i, :, :, layer].cpu()), cmap="gray")
    
    plt.tight_layout()
    plt.show()