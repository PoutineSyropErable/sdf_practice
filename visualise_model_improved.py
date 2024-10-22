import numpy as np
import torch
import torch.nn as nn
import polyscope as ps
from skimage.measure import marching_cubes

# Define your model architecture as done previously
from train import SimpleNN

def load_model(file_name: str = 'point_distance_model.pth') -> SimpleNN:
    model = SimpleNN()  
    model.load_state_dict(torch.load(file_name, weights_only=True))
    model.eval()  
    return model

def create_grid(bounds: np.ndarray, grid_size: int) -> np.ndarray:
    """Create a grid of points within the specified bounds."""
    x = np.linspace(bounds[0][0], bounds[0][1], grid_size)
    y = np.linspace(bounds[1][0], bounds[1][1], grid_size)
    z = np.linspace(bounds[2][0], bounds[2][1], grid_size)
    grid = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    return grid

def ml_sdf_array(vec3_array: np.ndarray, trained_model: nn.Module) -> np.ndarray:
    """Calculate signed distances for an array of points."""
    if vec3_array.shape[1] != 3:  # Ensure the shape is (N, 3)
        raise ValueError("Input must be a 2D NumPy array with shape (N, 3)")
    
    input_tensor = torch.tensor(vec3_array, dtype=torch.float32)
    with torch.no_grad():
        torch_output = trained_model(input_tensor)
    output = torch_output.detach().numpy()
    return output[:, 0]  # Assuming output shape is (N, 1)

def adaptive_grid(bounds: np.ndarray, initial_grid_size: int, trained_model: nn.Module, threshold: float, max_iterations: int) -> np.ndarray:
    """Create an adaptive grid based on SDF gradients."""
    grid_points = create_grid(bounds, initial_grid_size)
    sdf_values = ml_sdf_array(grid_points, trained_model)
    
    for _ in range(max_iterations):
        # Check if there are points with a significant change
        significant_changes = np.abs(sdf_values) < threshold
        if not np.any(significant_changes):
            break  # No more refinements needed
        
        # Refine grid around significant changes
        new_grid_size = initial_grid_size * 2  # Double the resolution
        refined_grid_points = create_grid(bounds, new_grid_size)
        refined_sdf_values = ml_sdf_array(refined_grid_points, trained_model)
        
        # Combine old and refined points
        grid_points = np.vstack((grid_points, refined_grid_points))
        sdf_values = np.concatenate((sdf_values, refined_sdf_values))
    
    return grid_points

def recreate_mesh_from_sdf(sdf_values: np.ndarray, grid_shape: tuple) -> tuple:
    """Apply Marching Cubes to extract the surface from SDF values."""
    vertices, faces, _, _ = marching_cubes(sdf_values.reshape(grid_shape), level=0)
    return vertices, faces

def main():
    # Load the trained model
    trained_model = load_model()

    # Define the bounding box with min and max values
    b_min = np.array([-0.21074983, -0.08195469, -0.15197453])  # min bounds
    b_max = np.array([0.17714821, 0.30226061, 0.14901])  # max bounds
    bounds = np.array([[b_min[0], b_max[0]], [b_min[1], b_max[1]], [b_min[2], b_max[2]]])
    
    # Set parameters for the adaptive grid
    initial_grid_size = 20  # Initial resolution
    threshold = 0.05  # Threshold for SDF value changes
    max_iterations = 3  # Max iterations for refinement

    # Create an adaptive grid of points within the bounding box
    grid_points = adaptive_grid(bounds, initial_grid_size, trained_model, threshold, max_iterations)

    # Calculate SDF values for the grid points
    sdf_values = ml_sdf_array(grid_points, trained_model)

    # Recreate the mesh from SDF values
    vertices, faces = recreate_mesh_from_sdf(sdf_values, (initial_grid_size * (2 ** max_iterations),) * 3)

    # Visualize the reconstructed mesh
    ps.init()
    ps_mesh = ps.register_surface_mesh("Reconstructed Model", vertices, faces)
    ps.show()

if __name__ == "__main__":
    main()
