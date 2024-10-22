import numpy as np 
import igl 

import igl

import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader, Dataset


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
        print("min, max=",min,max)

        center = (min_val + max_val) / 2
        half_length = max_val - center 

        BOX_RATIO = 1.5

        print("c,h=",center,half_length)

        b_min[i] = min_val - half_length * BOX_RATIO 
        b_max[i] = max_val + half_length * BOX_RATIO

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

    signed_distances, closest_face, three_closest = igl.signed_distance(point_list, V, F)
    return signed_distances, closest_face, three_closest







def weight_function( signed_distance : float, weight_exponent: float = 8) -> float:
    """ Takes a signed_distances and return a probability of taking said points"""
    return (1 + abs(signed_distance))**(-weight_exponent)



def filter_function( signed_distance: float) -> bool:
    "Returns a bool or not, deciding weither or not to take the point"

    random_number = np.random.rand()
    return random_number < weight_function(signed_distance) 

def filter_points(signed_distances: np.ndarray) -> np.ndarray:
    """Filter points based on their signed distances."""

    filtered_index = np.array([i for i in range(len(signed_distances)) if filter_function(signed_distances[i])])
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
   # Load the mesh
    V, F = load_mesh("bunny.obj")
    print("\nV=", V, "\n\n")

    # Compute bounding box
    b_min, b_max = compute_bounding_box(V)
    print("------\n\n\n")
    print(f"b_min = {b_min},  b_max ={b_max}\n\n\n")

    # Generate random points
    NUMBER_POINT = 100_000
    point_list = generate_random_points(b_min, b_max, NUMBER_POINT)

    print("\n\n")
    print(f"point_list=\n{point_list}")

    # Compute signed distances
    signed_distances, closest_face, three_closest = compute_signed_distances(point_list, V, F)
    print(f"\n\ndistance = {signed_distances}\n")
    print(f"closest_face = {closest_face}\n")
    print(f"three_closest = {three_closest}\n")

    # Print min/max signed distances
    min_dn, max_dn = np.min(signed_distances[signed_distances < 0]), np.max(signed_distances[signed_distances < 0])
    min_dp, max_dp = np.min(signed_distances[signed_distances > 0]), np.max(signed_distances[signed_distances > 0])
    print(f"min_dn={min_dn}, max_dn={max_dn}")
    print(f"min_dp={min_dp}, max_dp={max_dp}")

    # Filter points
    filtered_index = filter_points(signed_distances)
    filtered_signed_distances = signed_distances[filtered_index]
    filtered_points = point_list[filtered_index]

    print(f"\n\n\nfiltered_index= {filtered_index}, {len(filtered_index)} \n")
    print(f"filtered_points = \n{filtered_points} \n")
    print(f"filtered_signed_distances = \n{filtered_signed_distances} \n")


    #Histogram?
    show_histogram(filtered_signed_distances)

    # Doing the machine learning steps
    print(f"\n\n------------------\nRuns on GPU?: {torch.cuda.is_available()}\n------------------\n\n")
    data_loader = prepare_data(filtered_points, filtered_signed_distances)
    trained_model = train_model(data_loader, num_epochs=100, learning_rate=0.001)
    save_model(trained_model)

    



if __name__ == "__main__":
    main()










