import numpy as np 
import igl 
import polyscope as ps

import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader, Dataset

import os, sys 


DEFAULT_NUM_EPOCHS = 10


def load_mesh(file_path: str):
    """Load a triangle mesh from a given file path."""

    V, F = igl.read_triangle_mesh(file_path)
    return V, F



def compute_bounding_box(V: np.ndarray) -> (np.ndarray, np.ndarray):
    """Compute the bounding box for the vertices."""
    b_min = np.zeros(3)
    b_max = np.zeros(3)

    for i in range(3):
        min_val, max_val = V[:, i].min(), V[:, i].max()
        print("min, max=",min_val,max_val)

        center = (min_val + max_val) / 2
        half_length = max_val - center 

        BOX_RATIO = 1.35

        print("c,h=",center,half_length)

        b_min[i] = min_val - half_length * BOX_RATIO 
        b_max[i] = max_val + half_length * BOX_RATIO

    return b_min, b_max


def compute_small_bounding_box(V: np.ndarray) -> (np.ndarray, np.ndarray):
    b_min = np.zeros(3)
    b_max = np.zeros(3)

    for i in range(3):
        min_val, max_val = V[:, i].min(), V[:, i].max()
        b_min[i] = min_val 
        b_max[i] = max_val 

    return b_min, b_max

    



    



def generate_random_points(b_min: np.ndarray, b_max: np.ndarray, num_points: int) -> np.ndarray:
    """Generate random points within the bounding box."""
    point_list = np.zeros(shape=(num_points, 3))
    
    # Working with one column at a time. IE: All X -> All Y -> All Z
    for i in range(3):
        random_points_i = np.random.uniform(b_min[i], b_max[i], num_points)
        print(f"rand_{i}=", random_points_i)
        point_list[:, i] = random_points_i 
        print("---")

    return point_list


def compute_signed_distances(point_list: np.ndarray, V: np.ndarray, F: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """Compute signed distances from points to the triangle mesh."""

    signed_distances, nearest_face, nearest_points = igl.signed_distance(point_list, V, F)
    return signed_distances, nearest_face, nearest_points


def draw_bounding_box(b_min: np.ndarray, b_max: np.ndarray, name: str, color: tuple = (0.0, 1.0, 0.0), radius: float = 0.002):
    """Draw a bounding box in Polyscope given min and max points."""
    # Create corners of the bounding box
    box_corners = np.array([[b_min[0], b_min[1], b_min[2]],
                            [b_max[0], b_min[1], b_min[2]],
                            [b_max[0], b_max[1], b_min[2]],
                            [b_min[0], b_max[1], b_min[2]],
                            [b_min[0], b_min[1], b_max[2]],
                            [b_max[0], b_min[1], b_max[2]],
                            [b_max[0], b_max[1], b_max[2]],
                            [b_min[0], b_max[1], b_max[2]]])

    # Define edges for the bounding box
    box_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                          [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                          [0, 4], [1, 5], [2, 6], [3, 7]])  # Vertical edges

    # Register the bounding box as a curve network
    ps_bounding_box = ps.register_curve_network(name, box_corners, box_edges)
    ps_bounding_box.set_radius(radius)  # Adjust bounding box line thickness
    ps_bounding_box.set_color(color)  # Set the color for the bounding box



def show_result_in_polyscope(V, F, filtered_points, filtered_signed_distances, filtered_nearest, ML_signed_distances ):
    ps.init()
    ps_mesh = ps.register_surface_mesh("Bunny", V, F)

    
    NUMBER_OF_POINTS = 20_000
    ps_cloud = ps.register_point_cloud("Flitered Points", filtered_points[0:NUMBER_OF_POINTS], radius=0.0025)
    ps_cloud.add_scalar_quantity("Signed Distances",filtered_signed_distances[0:NUMBER_OF_POINTS])
    ps_cloud.add_scalar_quantity("ML Signed Distances",ML_signed_distances[0:NUMBER_OF_POINTS])

    NUMBER_OF_LINES = 100
    # Create edges for the curve network
    edges = np.column_stack((np.arange(NUMBER_OF_LINES), np.arange(NUMBER_OF_LINES)))  # Connecting filtered points to their nearest points

    # Combine filtered points and filtered nearest points into a single array
    all_points = np.vstack((filtered_points[0:NUMBER_OF_LINES], filtered_nearest[0:NUMBER_OF_LINES]))
    print(f"\n\n\nall_points = \n{all_points}\nShape={np.shape(all_points)}")

    # Create edges that connect filtered_points to filtered_nearest
    edges = np.column_stack((np.arange(NUMBER_OF_LINES), np.arange(NUMBER_OF_LINES) + NUMBER_OF_LINES))  # Adjust edges for the combined array
    print(f"\n\n\nedges = \n{edges}\nShape={np.shape(edges)}")


    # Register the curve network to show lines from filtered_points to filtered_nearest
    ps_lines = ps.register_curve_network("Lines to Nearest Points", all_points, edges)

    # Optional: Customize appearance of the lines
    ps_lines.set_radius(0.001)  # Adjust line thickness
    ps_lines.set_color((1.0, 0.0, 0.0))  # Red color for the lines

     # Compute and draw the larger bounding box
    b_min, b_max = compute_bounding_box(V)
    draw_bounding_box(b_min, b_max, "Large Bounding Box", color=(0.0, 1.0, 0.0), radius=0.002)

    # Compute and draw the smaller bounding box
    small_b_min, small_b_max = compute_small_bounding_box(V)
    draw_bounding_box(small_b_min, small_b_max, "Small Bounding Box", color=(0.0, 0.0, 1.0), radius=0.001)

    ps.show()




def weight_function( signed_distance : float, weight_exponent: float = 8) -> float:
    """ Takes a signed_distances and return a probability of taking said points"""
    return (1 + abs(signed_distance))**(-weight_exponent)



def filter_function( signed_distance: float, weight_exponent: float) -> bool:
    "Returns a bool or not, deciding weither or not to take the point"

    random_number = np.random.rand()
    return random_number < weight_function(signed_distance, weight_exponent) 

def filter_points(signed_distances: np.ndarray, weight_exponent : float) -> np.ndarray:
    """Filter points based on their signed distances."""

    filtered_index = np.array([i for i in range(len(signed_distances)) if filter_function(signed_distances[i], weight_exponent)])
    return filtered_index



def show_histogram(filtered_signed_distances):
    # Show Histogram of signed_distances
    plt.figure()
    plt.title("Histogram of Signed Distances")
    plt.grid()
    plt.xlabel("Signed Distances")
    plt.ylabel("Counts")
    plt.hist(filtered_signed_distances)
    plt.show()



#---------------------- Start of ML ----------------------

# Step 1: Create a custom Dataset class
class PointDistanceDataset(Dataset):
    def __init__(self, points: np.ndarray, distances: np.ndarray):
        self.points = torch.tensor(points, dtype=torch.float32)
        self.distances = torch.tensor(distances, dtype=torch.float32)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx], self.distances[idx]

# Step 2: Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # 3 input features for the 3D points
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)   # 1 output feature for the signed distance

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def prepare_data(filtered_points: np.ndarray, filtered_signed_distances: np.ndarray, batch_size: int = 32) -> DataLoader:
    """Prepare the data loader for training."""
    dataset = PointDistanceDataset(filtered_points, filtered_signed_distances)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_model(data_loader: DataLoader, num_epochs: int = 100, learning_rate: float = 0.001) -> SimpleNN:
    """Train the model on the provided data loader."""
    model = SimpleNN()
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()         # Zero the gradients
            outputs = model(inputs)      # Forward pass
            loss = criterion(outputs.squeeze(), targets)  # Compute the loss
            loss.backward()               # Backward pass
            optimizer.step()              # Update weights

        # Print loss every 10 epochs (I want to see epoch 0)
        if (epoch +1 ) % 10 == 0 or epoch == 1:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.16f}')

    return model

def save_model(model: SimpleNN, file_name: str = 'point_distance_model.pth') -> None:
    """Save the trained model to a file."""
    torch.save(model.state_dict(), file_name)
    print(f"Model saved to {file_name}")


def main():

    print(f"\n\n{'-'*10} Start of Program{'-'*10}\n\n")
    os.chdir(sys.path[0])
    print(f"Current Path = {os.getcwd()}\n")
    args = sys.argv

    # First argument represent the number of epochs, if no argument given, go to 100
    if len(args) > 1:
        try:
            NUMBER_EPOCHS = int(args[1])  # Parse the first argument as an integer
        except ValueError:
            print("Invalid input for epochs. Using default value: 100.")
            NUMBER_EPOCHS = DEFAULT_NUM_EPOCHS
    else:
        NUMBER_EPOCHS = DEFAULT_NUM_EPOCHS  # Default value

    print(f"Number of EPOCHS: {NUMBER_EPOCHS}\n")

   # Load the mesh
    V, F = load_mesh("bunny.obj")
    print("\nV=", V, "\n\n")

    # Compute bounding box
    b_min, b_max = compute_bounding_box(V)
    print("------\n\n\n")
    print(f"b_min = {b_min},  b_max ={b_max}\n\n\n")

    # Generate random points
    NUMBER_POINT = 1_000_000
    point_list = generate_random_points(b_min, b_max, NUMBER_POINT)

    print("\n\n")
    print(f"point_list=\n{point_list}")

    # Compute signed distances
    signed_distances, nearest_face, nearest_points = compute_signed_distances(point_list, V, F)

    diff = nearest_points - point_list 
    l2_norms = np.linalg.norm(diff, axis=1)

    print(f"\n\n  signed distance   = {signed_distances}")
    print(f"Norm(nearest-point) = {l2_norms}\n")
    print(f"nearest_face = {nearest_face}\n")
    print(f"nearest_points = {nearest_points}\n")
    print(f"3 nearest_shape = {np.shape(nearest_points)}")



    # Print min/max signed distances
    min_dn, max_dn = np.min(signed_distances[signed_distances < 0]), np.max(signed_distances[signed_distances < 0])
    min_dp, max_dp = np.min(signed_distances[signed_distances > 0]), np.max(signed_distances[signed_distances > 0])
    print(f"min_dn={min_dn}, max_dn={max_dn}")
    print(f"min_dp={min_dp}, max_dp={max_dp}")

    

    # Filter points
    filtered_index = filter_points(signed_distances, weight_exponent = 12)
    filtered_signed_distances = signed_distances[filtered_index]
    filtered_points = point_list[filtered_index]
    filtered_nearest = nearest_points[filtered_index]

    print(f"\n\n\nfiltered_index= {filtered_index}, {len(filtered_index)} \n")
    print(f"filtered_points = \n{filtered_points} \n")
    print(f"filtered_signed_distances = \n{filtered_signed_distances} \n")
    

    #Histogram?
    # show_histogram(filtered_signed_distances)



    
    
    
    # Doing the machine learning steps
    print(f"\n\n------------------\nRuns on GPU?: {torch.cuda.is_available()}\n------------------\n\n")
    data_loader = prepare_data(filtered_points, filtered_signed_distances)
    trained_model = train_model(data_loader, num_epochs=NUMBER_EPOCHS, learning_rate=0.001)

    file_name = 'point_distance_model_'+str(NUMBER_EPOCHS)+'.pth'
    save_model(trained_model,file_name)


    def ml_sdf(vec3: np.ndarray) -> float:
        if vec3.shape != (3,):  # Check if the shape is (3,)
            raise ValueError("Input must be a 1D NumPy array of size 3")
            
        input_tensor = torch.from_numpy(vec3).float()  # Create a tensor from the NumPy array
        torch_output = trained_model(input_tensor.unsqueeze(0))  # Add a batch dimension
        output = torch_output.detach().numpy()[0]
        
        print(f"output = {output}\n")
        return output

    def ml_sdf_array(vec3_array: np.ndarray) -> np.ndarray:
        if vec3_array.shape[1] != 3:  # Check if the shape is (N, 3)
            raise ValueError("Input must be a 2D NumPy array with shape (N, 3)")
        
        input_tensor = torch.from_numpy(vec3_array).float()  # Create a tensor from the NumPy array
        torch_output = trained_model(input_tensor)  # No need for unsqueeze as it's already 2D
        output = torch_output.detach().numpy()[:, 0]  # Assuming output shape is (N, 1)
        
        print(f"output = {output}\n")
        return output




    ml_sdf(filtered_points[0])


    #Do calculation with the model
    ML_signed_distances = ml_sdf_array(filtered_points) 
    print(f"ML_signed_distances = \n{ML_signed_distances}\n")

    

    show_result_in_polyscope(V,F, filtered_points, filtered_signed_distances, filtered_nearest, ML_signed_distances )


    



if __name__ == "__main__":
    main()








