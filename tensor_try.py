import torch
import numpy as np

# create tensor from data
data = [[0 , 1, 2],[3, 4 , 5]]
x_data = torch.tensor(data)
print(f"x_data\n {x_data}")

# create tensor from numpy
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"x_np\n {x_np}")