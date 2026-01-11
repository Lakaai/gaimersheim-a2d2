import json
from dataclasses import dataclass

import numpy as np

with open("camera.json", "r") as f:
    calibration_data = json.load(f)

camera_matrix = np.array(calibration_data["camera_matrix"], dtype=np.float64)
dist_coeffs = np.array(calibration_data["dist_coeffs"], dtype=np.float64)


@dataclass
class Camera:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    horizontal_fov: float
    vertical_fov: float
    diagonal_fov: float


@dataclass
class Pose:
    something: np.ndarray


@dataclass
class ChessboardImage:
    image_path: str
    image: np.ndarray
    pose: Pose
    corners: np.ndarray


def draw_box():
    box_height = -0.23  # Height of the box in meters
    number_of_segments = 0


def recover_pose():
    """Recover the extrisic parameters (camera pose) for each chessboard image."""
