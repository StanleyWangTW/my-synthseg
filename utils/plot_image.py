import matplotlib.pyplot as plt

def show_slices(image, layer, cmap):
    plt.figure(figsize=(10, 10))

    plt.subplot(131)
    plt.title("Sagittal")
    plt.xlabel(f"Layer: {layer[0]}")
    plt.imshow(image[layer[0], :, :], cmap=cmap)

    plt.subplot(132)
    plt.title("Coronal")
    plt.xlabel(f"Layer: {layer[1]}")
    plt.imshow(image[:, layer[1], :], cmap=cmap)

    plt.subplot(133)
    plt.title("Axial")
    plt.xlabel(f"Layer: {layer[2]}")
    plt.imshow(image[:, :, layer[2]], cmap=cmap)

    plt.show()