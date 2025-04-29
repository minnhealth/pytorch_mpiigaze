import dataclasses
from typing import Optional

import cv2
import numpy as np
import yaml

@dataclasses.dataclass()
class Camera:
    width: int = dataclasses.field(init=False)
    height: int = dataclasses.field(init=False)
    camera_matrix: np.ndarray = dataclasses.field(init=False)
    dist_coefficients: np.ndarray = dataclasses.field(init=False)

    camera_params_path: dataclasses.InitVar[str] = None

    def __post_init__(self, camera_params_path):
        with open(camera_params_path) as f:
            data = yaml.safe_load(f)

        self.width = int(data['image_width'])
        self.height = int(data['image_height'])

        self.camera_matrix = np.array(data['camera_matrix']['data'], dtype=np.float32).reshape(3, 3)
        self.dist_coefficients = np.array(data['distortion_coefficients']['data'], dtype=np.float32).reshape(-1, 1)

        # ✅ Extra safety checks
        assert self.camera_matrix.shape == (3, 3), f"camera_matrix shape invalid: {self.camera_matrix.shape}"
        assert self.dist_coefficients.ndim == 2, f"dist_coefficients ndim invalid: {self.dist_coefficients.ndim}"

    def project_points(self,
                       points3d: np.ndarray,
                       rvec: Optional[np.ndarray] = None,
                       tvec: Optional[np.ndarray] = None) -> np.ndarray:
        assert points3d.shape[1] == 3
        if rvec is None:
            rvec = np.zeros(3, dtype=np.float32)
        if tvec is None:
            tvec = np.zeros(3, dtype=np.float32)
        points2d, _ = cv2.projectPoints(points3d, rvec, tvec,
                                        self.camera_matrix,
                                        self.dist_coefficients)
        return points2d.reshape(-1, 2)