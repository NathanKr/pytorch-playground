import torch
import numpy as np

# pylint give warning , switch to e.g. flake8

# create tensor from data
data = [[0, 1, 2], [3, 4, 5]]
t_data = torch.tensor(data)
print(f"t_data\n {t_data}")


# create tensor from numpy
np_array = np.array(data)
t_np = torch.from_numpy(np_array)
print(f"t_np\n {t_np}")

# create tensor from another tensor
t_ones = torch.ones_like(t_data)
print(f"t_ones\n {t_ones}")

# not using dtype uses the source data type
t_rand = torch.rand_like(t_data, dtype=torch.float)
print(f"t_rand\n {t_rand}")


# use shape
s = (2, 3)
t_rand_from_shape = torch.rand(s)
print(f"t_rand_from_shape\n {t_rand_from_shape}")

t_ones_from_shape = torch.ones(s)
print(f"t_ones_from_shape\n {t_ones_from_shape}")


t_zeros_from_shape = torch.zeros(s)
print(f"t_zeros_from_shape\n {t_zeros_from_shape}")


# tensor attributes
print(f"t_rand.shape : {t_rand.shape}")
print(f"t_rand.dtype : {t_rand.dtype}")
print(f"t_rand.device : {t_rand.device}")

# ********** tensor operations

# this issues warning about Found no NVIDIA driver on your system
if torch.cuda.is_available():
    print("cuda is available")
else:
    print("cuda is not available")


# numpy-like indexing and slicing
t = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"t\n{t}")
print(f"first raw\n{t[0,:]}")
print(f"first column\n{t[:,0]}")

# concat tensor
t1 = torch.cat([t, t], dim=0)
print("concat on rows\n", t1)

t1 = torch.cat([t, t], dim=1)
print("concat on columns\n", t1)


# tensors element by element multiplication
t = torch.ones((3, 3))
t1 = t * t
print("multiply element by element :  t * t", t1)
t1 = t.mul(t)
print("multiply element by element :  t.mul(t)", t1)
t1 = t.mul(t)

# tensors multiplication
t = torch.ones((3, 3))
t1 = t.matmul(t)
print("multiply tensors :  t.matmul(t)", t1)
t1 = t @ t
print("multiply tensors :  t @ t", t1)


# ********** bridge with numpy - share same memory (pointer like)
t = torch.ones(5)
print("torch.ones(5)\n", t)

n = t.numpy()
print("t.numpy()\n", n)
t.add_(1)
print("t after adding 1\n", t)
print("n after adding 1 to t\n", n)

t1 = torch.from_numpy(n)
print(f"t1 : created from numpy\n{t1}")

