import numpy as np
import igl
import polyscope as ps
import time

# Function to calculate the signed distance function (SDF)
def compute_sdf(points, mesh_vertices, mesh_faces):
    # Here we assume you have a function that calculates the SDF
    # This could be a placeholder for the actual SDF computation
    sdt = np.random.rand(len(points))  # Replace with actual SDF computation
    return sdt

# Function to generate a grid of points
def generate_grid(xmin, xmax, ymin, ymax, zmin, zmax, N, M, L, rx, ry, rz):
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    z_center = (zmin + zmax) / 2

    x_span = (xmax - xmin) * (1 + rx)
    y_span = (ymax - ymin) * (1 + ry)
    z_span = (zmax - zmin) * (1 + rz)

    x_points = np.linspace(x_center - x_span / 2, x_center + x_span / 2, N)
    y_points = np.linspace(y_center - y_span / 2, y_center + y_span / 2, M)
    z_points = np.linspace(z_center - z_span / 2, z_center + z_span / 2, L)

    grid_points = np.array(np.meshgrid(x_points, y_points, z_points)).T.reshape(-1, 3)
    return grid_points

# Click callback function
def point_click_callback(idx):
    point_info = grid_points[idx]  # Get the coordinates of the clicked point
    sdt_value = sdt[idx]            # Get the corresponding SDF value
    print(f"Point clicked: {point_info}, SDF: {sdt_value}")

# Main execution block
if __name__ == "__main__":
    start_time = time.time()

    # Load the mesh
    mesh_vertices, mesh_faces = igl.read_triangle_mesh("bunny.obj")

    # Calculate bounding box
    bbox_min = np.min(mesh_vertices, axis=0)
    bbox_max = np.max(mesh_vertices, axis=0)
    print(f"Bounding box min: {bbox_min}, max: {bbox_max}")

    # Generate points in a grid
    N, M, L = 10, 10, 10  # Change as needed
    rx, ry, rz = 0.1, 0.1, 0.1  # Modify to control the size of the grid
    grid_points = generate_grid(bbox_min[0], bbox_max[0],
                                 bbox_min[1], bbox_max[1],
                                 bbox_min[2], bbox_max[2],
                                 N, M, L, rx, ry, rz)

    # Compute SDF for the grid points
    sdt = compute_sdf(grid_points, mesh_vertices, mesh_faces)
    print("Signed Distance Function (SDF):", sdt)

    # Initialize Polyscope
    ps.init()

    # Add the bunny mesh
    ps_mesh = ps.Mesh("Bunny", mesh_vertices, mesh_faces)
    ps_mesh.add_vertex_color("sdf", sdt)  # Visualize SDF as vertex colors
    ps_mesh.add_edge_color("black")  # Optional: add edge color
    ps_mesh.set_vertex_radius(0.01)  # Set vertex size
    ps_mesh.set_edge_radius(0.005)  # Set edge size

    # Add the bounding box as a mesh
    bbox_vertices = np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_max[1], bbox_max[2]],
        [bbox_min[0], bbox_max[1], bbox_max[2]],
    ])
    
    bbox_faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom
        [4, 5, 6], [4, 6, 7],  # Top
        [0, 1, 5], [0, 5, 4],  # Side
        [1, 2, 6], [1, 6, 5],  # Side
        [2, 3, 7], [2, 7, 6],  # Side
        [3, 0, 4], [3, 4, 7],  # Side
    ])
    
    ps_bbox = ps.Mesh("Bounding Box", bbox_vertices, bbox_faces)
    ps_bbox.set_color("blue")  # Set bounding box color

    # Add grid points as interactive point cloud
    ps_points = ps.PointCloud("Grid Points", grid_points)
    ps_points.set_radius(0.01)  # Set size of the grid points
    ps_points.set_color("red")  # Set color of the grid points
    
    # Register the point click callback
    ps_points.set_click_callback(point_click_callback)

    # Show the visualization
    ps.show()

    # Measure elapsed time
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

