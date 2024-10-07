import igl
import numpy as np
import time

def compute_aabb(V):
    """
    Compute the Axis-Aligned Bounding Box (AABB) of the vertices.
    """
    min_corner = V.min(axis=0)
    max_corner = V.max(axis=0)
    return min_corner, max_corner

def generate_grid(min_corner, max_corner, N, M, L, rx=1.0, ry=1.0, rz=1.0):
    """
    Generate a grid of points inside the expanded bounding box.
    
    The grid will have NxMxL points in the x, y, and z directions.
    The expansion is determined by the rx, ry, and rz parameters.
    """
    # Compute the center of the box in each dimension
    x_center = (min_corner[0] + max_corner[0]) / 2.0
    y_center = (min_corner[1] + max_corner[1]) / 2.0
    z_center = (min_corner[2] + max_corner[2]) / 2.0

    # Calculate the half-size of the bounding box
    x_half_size = (max_corner[0] - min_corner[0]) / 2.0
    y_half_size = (max_corner[1] - min_corner[1]) / 2.0
    z_half_size = (max_corner[2] - min_corner[2]) / 2.0

    # Adjust the half-size by the rx, ry, rz scaling factors
    x_half_size *= rx
    y_half_size *= ry
    z_half_size *= rz

    # Define the new min and max corners based on the expanded size
    new_min_corner = np.array([x_center - x_half_size, y_center - y_half_size, z_center - z_half_size])
    new_max_corner = np.array([x_center + x_half_size, y_center + y_half_size, z_center + z_half_size])

    # Generate the grid of points
    x = np.linspace(new_min_corner[0], new_max_corner[0], N)
    y = np.linspace(new_min_corner[1], new_max_corner[1], M)
    z = np.linspace(new_min_corner[2], new_max_corner[2], L)

    # Create a 3D mesh grid of points
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Flatten the grid into a list of 3D points
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    return grid_points

def compute_signed_distances(V, F, points):
    """
    Compute the signed distance from the given points to the surface of the mesh.
    """
    # Compute signed distance
    S, I, C = igl.signed_distance(points, V, F, sign_type=igl.SIGNED_DISTANCE_TYPE_DEFAULT)
    return S

def main():
    # Load the bunny mesh
    mesh_path = "bunny.obj"  # Adjust this to the correct path
    V, F = igl.read_triangle_mesh(mesh_path)

    # Compute the AABB
    min_corner, max_corner = compute_aabb(V)

    # Generate a grid of points with expansion factors (example: 10x10x10 grid with expansion)
    N, M, L = 10, 10, 10  # You can adjust these values
    rx, ry, rz = 1.5, 1.5, 1.5  # Example expansion factors, adjust these as needed
    grid_points = generate_grid(min_corner, max_corner, N, M, L, rx, ry, rz)

    # Measure the time it takes to compute signed distances
    start_time = time.time()

    # Compute signed distance to the bunny surface for each grid point
    signed_distances = compute_signed_distances(V, F, grid_points)

    end_time = time.time()

    # Print the signed distances and the time taken
    print("Signed Distances:")
    print(signed_distances)

    print(f"Time taken: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main()

