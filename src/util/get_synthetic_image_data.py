from pydantic import BaseModel, root_validator
from pathlib import Path
from PIL import Image
import numpy as np
import json

class SyntheticFrame(BaseModel):
    file_path: str
    rotation: float
    transform_matrix: list[list[float]]
    data_path : Path = None

    @property
    def image_path(self):
        return str((self.data_path / self.file_path).resolve()) + '.png'
    
    @property
    def image(self):
        return Image.open(self.image_path)
    
    @property
    def depth_image_path(self):
        # only works for test datasets!
        return str((self.data_path / self.file_path).resolve()) + '_depth_0001.png'

    @property
    def matrix_np(self):
        return np.array(self.transform_matrix)

class PoseData(BaseModel):
    camera_angle_x: float
    frames: list[SyntheticFrame]
    data_path : Path

    @root_validator(pre=True)
    def set_data_path(cls, values):
        data_path = values.get('data_path')
        frames = values.get('frames')
        if data_path and frames:
            for frame in frames:
                frame['data_path'] = data_path
        return values

def get_image_data(data_path, mode='train'):  
    data_path = Path(data_path)
    with open(data_path/f"transforms_{mode}.json", 'r') as f:
        json_data = json.load(f)

    json_data['data_path'] = data_path
    return PoseData.parse_obj(json_data)
