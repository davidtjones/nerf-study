import matplotlib.pyplot as plt
import numpy as np
from util.data import NeRFDataset
import torch
import code


def test_dataset_chunking():
    """
    This test is designed to show that chunking builds the same image as no 
    chunking. 
    """
    data_path = "../data/nerf_synthetic/lego"

    base_image_size = 800
    test_chunk_size = 200  # this works even with weird numbers

    test_chunks_in_base = int(np.ceil(base_image_size / test_chunk_size) ** 2)

    base_dataset = NeRFDataset(
        data_path,
        mode='train',
        chunk_size=base_image_size
    )

    test_dataset = NeRFDataset(
        data_path,
        mode='train',
        chunk_size=test_chunk_size
    )

    img_chunks = []

    for i in range(test_chunks_in_base):
        _, img_chunk = test_dataset[i]
        img_chunks.append(img_chunk)

    _, base_image = base_dataset[0]


    img_recon = test_dataset.image_from_chunks(img_chunks)

    print(img_recon.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img_recon.numpy())
    ax2.imshow(base_image.numpy())
    plt.savefig("reconstruction.png")
    plt.close()

    assert torch.allclose(img_recon, base_image)
