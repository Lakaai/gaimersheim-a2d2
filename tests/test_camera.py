import numpy as np

from src.camera import vector_to_pixel, pixel_to_vector, Camera

def test_pixel_to_vector():
    camera_matrix = np.array([[5.0, 0.0, 0.1],
                              [0.0, 6.0, 0.2],
                              [0.0, 0.0, 1.0]])
    dist_coeffs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    camera = Camera(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

    rQOi = np.array([5.3, 8.7])
    uPCc = pixel_to_vector(rQOi, camera)
    expected_uPCc = np.array([0.51433847, 0.70062124, 0.49455618])

    assert np.allclose(uPCc, expected_uPCc)

def test_vector_to_pixel():
    camera_matrix = np.array([[5.0, 0.0, 0.1],
                              [0.0, 6.0, 0.2],
                              [0.0, 0.0, 1.0]])
    dist_coeffs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    camera = Camera(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

    uPCc = np.array([5.3, 8.7])
    rQOi = vector_to_pixel(uPCc, camera)
    # expected_rQOi = np.array([5.3/5.0 - (camera.camera_matrix[2][2] * uPCc[2]) / (camera.camera_matrix[2][2] * uPCc[2]), 
    #                           (8.7/6.0 - (camera.camera_matrix[2][2] * uPCc[2]) / (camera.camera_matrix[2][2] * uPCc[2]))])

    # assert np.allclose(rQOi, expected_rQOi)