import numpy as np 
import igl 

import torch 
import torch.nn as nn
import torch.optim as optim 


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



if __name__ == "__main__":
    main()
