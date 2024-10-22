import numpy as np 
import igl 

import torch 
import torch.nn as nn
import torch.optim as optim 





V, F = igl.read_triangle_mesh("bunny.obj")

print("\nV=",V,"\n\n")

b_min = np.zeros(3)
b_max = np.zeros(3)
for i in range(3):
    min, max = V[:,i].min(), V[:,i].max()
    print("min, max=",min,max)

    center = (min+max)/2
    half_length = max - center 

    BOX_RATIO = 1.5

    print("c,h=",center,half_length)

    b_min[i] = min - half_length*BOX_RATIO 
    b_max[i] = max + half_length*BOX_RATIO
    print("---")


print("------\n\n\n")
print(f"b_min = {b_min},  b_max ={b_max}\n\n\n") 

NUMBER_POINT = 100_000
point_list = np.zeros(shape=(NUMBER_POINT,3))
for i in range(3):
    random_points_i = np.random.uniform(b_min[i], b_max[i], NUMBER_POINT)
    print(f"rand_{i}=",random_points_i)
    point_list[:,i] = random_points_i 
    print("---")


print("\n\n")
print(f"point_list=\n{point_list}")

signed_distances, closest_face, three_closest = igl.signed_distance(point_list, V,F)
print(f"\n\ndistance = {signed_distances}\n")
print(f"closest_face = {closest_face}\n")
print(f"three_closest = {three_closest}\n")


min_dn, max_dn = np.min(signed_distances[signed_distances < 0]), np.max(signed_distances[signed_distances < 0])
min_dp, max_dp = np.min(signed_distances[signed_distances > 0]), np.max(signed_distances[signed_distances > 0])

print(f"min_dn={min_dn}, max_dn={max_dn}")
print(f"min_dp={min_dp}, max_dp={max_dp}")


def weight_function( signed_distance : float, weight_exponent: float = 8) -> float:
    """ Takes a signed_distances and return a probability of taking said points"""
    return (1 + abs(signed_distance))**(-weight_exponent)



def filter_function( signed_distance: float) -> bool:
    "Returns a bool or not, deciding weither or not to take the point"
    random_number = np.random.rand()
    return random_number < weight_function(signed_distance) 




filtered_index = np.array([  i for i in range(len(signed_distances)) if filter_function(signed_distances[i])  ]) 

filtered_signed_distances = signed_distances[filtered_index]
filtered_points = point_list[filtered_index]



print(f"\n\n\nfiltered_index= {filtered_index}, {len(filtered_index)} \n")
print(f"filtered_points = \n{filtered_points} \n")
print(f"filtered_signed_distances = \n{filtered_signed_distances} \n")





