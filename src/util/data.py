from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from util.get_synthetic_image_data import get_image_data
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
import code

from pytorch_lightning import LightningDataModule

def get_dirs(
        height: int, 
        width: int, 
        focal_length: float, 
        device="cpu") -> torch.Tensor:
    """
    Obtain viewing direction for each pixel from camera center. (Pinhole camera
    model) to gather directions at each pixel.

    Separating this component from get_rays allows us to chunk the ray-building
    process more easily by passing slices of `dirs` to get_rays.

    """
    
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32).to(device),
        torch.arange(height, dtype=torch.float32).to(device), 
        indexing='xy')

    # Map x,y image coordinates to a new range such that the center of the 
    # image is roughly ~(0,0), the top left is ~(-W/2, H/2), the bottom 
    # right is ~(W/2, -H/2). Assume the camera is oriented along the -z 
    # axis (this is a standard convention).
    dirs = torch.stack([
        (i-width*.5) / focal_length,
        -(j-height*.5) / focal_length,
        -torch.ones_like(i)
        ], dim=-1)
    
    return dirs

def get_rays(
        dirs: torch.Tensor, 
        c2w: torch.Tensor,
        normalize: bool=False,
        flatten: bool=False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build rays from viewing directions at each pixel.

    normalize: normalizes ray direction vectors to unit mag.
    flatten: converts rays from [H, W, 3] to [H*W, 3]. Useful for building 
                dataloaders.
    """
    
    # Build rays as a function of t in world coordinates r(t) = o + td
    rays_d = (dirs @ c2w[:3, :3].T)

    if normalize:
        rays_d /= torch.linalg.norm(rays_d, dim=-1, keepdims=True)

    # Origin from c2w directly
    rays_o = c2w[:3, 3].expand(rays_d.shape)

    if flatten:
        rays_d = rays_d.view(-1, 3)
        rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


class Empty:
    def __call__(self, x):
        return x

class SimpleImageCache:
    """
    Chunking during NeRFDataset.__getitem__ results in repeated loading of the 
    same image. This class caches that image to memory until a new image is
    needed. 
    """
    def __init__(self, frames):
        self.frames = frames
        self.cache = {}
        self.loads = 0

    def get(self, key):
        if key not in self.cache:
            self.loads += 1
            self.cache.clear()
            self.cache[key] = self.frames[key].image
        return self.cache[key]



class NeRFDataset(Dataset):
    """
    NeRF Dataset. 

    `chunk_size`: Specifying a chunk size can help alleviate memory bottleneck
        issues. This value should be the desired side-length for the image 
        chunk, e.g., 10 for a 100-ray chunk. If chunk_size**2 > self.H*self.W,
        or `None`, the whole image is used at once.

    `random`: randomizes the order that elements are returned from `__getitem__`
    """
    # TODO: add an optional argument that allows for building a dataset where
    #       the rays and rgb values are independent of the underlying dataset.
    
    def __init__(self, data_path, mode, chunk_size=None, transform=None):

        self.data_path = data_path
        self.mode = mode
        self.chunk_size= chunk_size
        self.transform = transform

        data = get_image_data(data_path, mode=mode)
       
        self.camera_angle_x = data.camera_angle_x
        self.frames = data.frames
        
        # load a test image and get the focal length:
        test_image = np.array(self.frames[0].image)
        test_image = test_image.transpose(2, 0, 1)
        test_image = torch.from_numpy(test_image)

        if self.transform:
            test_image = self.transform(test_image)
            
        self.H, self.W = test_image.shape[1:3]
        self.focal = self.W / (2 * np.tan(.5 * self.camera_angle_x))

        self.dirs = get_dirs(self.H, self.W, self.focal)

        # FUTURE: Remove this: this is for synthetic data only
        self.near = 2.
        self.far = 6.

        # Calculate chunking information
        self.chunk_size = self.chunk_size if self.chunk_size else self.H

        self.H_chunks = int(np.ceil(self.H / self.chunk_size))
        self.W_chunks = int(np.ceil(self.W / self.chunk_size))

        self.img_chunks_total = self.H_chunks * self.W_chunks

        self.img_cache = SimpleImageCache(self.frames)
            

        print(f"{mode.capitalize()} Dataset Characterisitics:\n"
              f"\tH:{self.H}, W:{self.W}, F:{self.focal:2f}\n"
              f"\tNear: {self.near}, Far:{self.far}\n"
              f"\tDataset size: {len(self.frames)}\n")

        
    def __len__(self):
        return len(self.frames) * self.H_chunks * self.W_chunks

    
    def __getitem__(self, idx):
        # idx is the specific chunk we want to access, where there are 
        # `num_frames * H_chunks * W_chunks` chunks, or `H_chunks * W_chunks`
        # per image.

        image_idx = idx // self.img_chunks_total
        # print(image_idx, idx, self.img_chunks_total)

        # Model needs Rays and RGB pixel values for each ray
        img = np.array(self.img_cache.get(image_idx))[..., :3]  # remove alpha
        
        img = torch.from_numpy(img).to(dtype=torch.float32) / 255.
        pose = torch.from_numpy(self.frames[image_idx].matrix_np).to(dtype=torch.float32)

        if self.transform:  # likely expects CHW
            img = img.permute(2, 0, 1)
            img = self.transform(img)
            img = img.permute(1, 2, 0)

        # get the correct H and W chunking slices
        chunk_idx = idx % self.img_chunks_total
        h = (chunk_idx // self.H_chunks) * self.chunk_size
        w = (chunk_idx % self.H_chunks) * self.chunk_size

        rays_o, rays_d = get_rays(
            self.dirs[h:h+self.chunk_size, w:w+self.chunk_size],  # index this!
            pose, 
            normalize=True, flatten=True)
        
        img_chunk = img[h:h+self.chunk_size, w:w+self.chunk_size, ...]

                
        return torch.cat([rays_o, rays_d], dim=-1), img_chunk
    
    def image_from_chunks(self, chunks):
        """
        Given a list a chunks (in the order of that they were produced from this
        class), reconstruct the full image. Returns a tensor.
        """
        rows = [torch.cat([chunks[j+i] for i in range(self.W_chunks)], dim=1)
                for j in range(0, self.img_chunks_total, self.H_chunks)]
        
        return torch.cat(rows)
    

class NeRFDataModule(LightningDataModule):
    def __init__(self, data_path, batch_size, transforms=None) -> None:
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.transforms = transforms

    def prepare_data(self) -> None:
        pass
    
    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = NeRFDataset(
                self.data_path, 
                mode="train", 
                transform=self.transforms,
                chunk_size=50
            )
            
            self.val_dataset = NeRFDataset(
                self.data_path,
                mode="val",
                transform=self.transforms,
                chunk_size=50
            )

        if stage == "test":
            self.test_dataset = NeRFDataset(
                self.data_path,
                mode="test",
                transform=self.transforms,
                chunk_size=50
            )
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=8, 
            shuffle=False)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=8, 
            shuffle=False)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=8, 
            shuffle=False)


