import polyscope as ps
import trimesh
import os, sys
import numpy as np
import igl

def show_mesh(file_path: str):
    # Start polyscope
    ps.init()

    # Load mesh using trimesh
    mesh = trimesh.load_mesh(file_path)
    
    # Ensure the file is successfully loaded
    if mesh.is_empty:
        print(f"Failed to load mesh from {file_path}")
        return

    # Register the mesh with polyscope
    ps.register_surface_mesh(
        "Mesh Viewer",
        mesh.vertices, 
        mesh.faces
    )

    # Show the mesh in the viewer
    ps.show()

# Example usage:
# show_mesh('relative/path/to/your/file.obj')

def main():
    # Check if a file path is provided as an argument
    if len(sys.argv) < 2:
        print("Usage: python script.py <relative/path/to/file.obj>")
        return

    # Get the file path from the first argument
    file_path = sys.argv[1]

    # Show the mesh using the provided file path
    show_mesh(file_path)

if __name__ == "__main__":
    main()


