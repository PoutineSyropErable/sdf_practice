import numpy as np
import torch
import polyscope as ps
from skimage.measure import marching_cubes

# Import your trained model
from train import SimpleNN

# Function to load the Model
def load_model(file_name: str = 'point_distance_model.pth') -> SimpleNN:
    model = SimpleNN()  
    model.load_state_dict(torch.load(file_name, weights_only=True))
    model.eval()  
    return model

NUM_EPOCH = 10
file_name = "point_distance_model_"+str(NUM_EPOCH)+".pth"
print("filename = ",file_name)
trained_model = load_model(file_name)


# Function to calculate signed distance for an array of points
def ml_sdf_array(vec3_array: np.ndarray) -> np.ndarray:
    if vec3_array.shape[1] != 3:  # Check if the shape is (N, 3)
        raise ValueError("Input must be a 2D NumPy array with shape (N, 3)")
    
    input_tensor = torch.from_numpy(vec3_array).float()  # Create a tensor from the NumPy array
    torch_output = trained_model(input_tensor)  # No need for unsqueeze as it's already 2D
    output = torch_output.detach().numpy()[:, 0]  # Assuming output shape is (N, 1)
    
    return output

# Function to create a 3D grid of points
def create_grid(bounds: np.ndarray, grid_size: int) -> np.ndarray:
    x = np.linspace(bounds[0, 0], bounds[0, 1], grid_size)
    y = np.linspace(bounds[1, 0], bounds[1, 1], grid_size)
    z = np.linspace(bounds[2, 0], bounds[2, 1], grid_size)
    return np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)  # Shape (N, 3)

# Function to recreate the mesh from SDF values
def recreate_mesh_from_sdf(sdf_values: np.ndarray, grid_shape: tuple):
    vertices, faces, _, _ = marching_cubes(sdf_values.reshape(grid_shape), level=0)
    return vertices, faces

def main():
    # Load the trained model
 

    # Define the bounding box with min and max values
    b_min = np.array([-0.21074983, -0.08195469, -0.15197453])  # min bounds
    b_max = np.array([0.17714821, 0.30226061, 0.14901])  # max bounds
    bounds = np.array([[b_min[0], b_max[0]], [b_min[1], b_max[1]], [b_min[2], b_max[2]]])
    grid_size = 200  # Increase resolution for better details

    # Create a 3D grid of points within the bounding box
    grid_points = create_grid(bounds, grid_size)

    # Calculate SDF values for the grid points
    sdf_values = ml_sdf_array(grid_points)

    # Recreate the mesh from SDF values
    vertices, faces = recreate_mesh_from_sdf(sdf_values, (grid_size, grid_size, grid_size))

    # Visualize the reconstructed mesh
    ps.init()
    ps_mesh = ps.register_surface_mesh("Reconstructed Model", vertices, faces)
    ps.show()

if __name__ == "__main__":
    main()

