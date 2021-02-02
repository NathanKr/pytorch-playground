import torch
# we need the derivative of the cost with respect to t1 and t2
# d_cost to d_t1 is 9 * t1 ** 2
# d_cost to d_t2 is -2 * t2

# doing the derivatives using backward --> the result is put in t1,t2

# option 1
t1_a = torch.tensor([2., 3.], requires_grad=True)
t2_a = torch.tensor([6., 4.], requires_grad=True)
cost_a = 3*t1_a**3 - t2_a**2

cost_a.sum().backward()
print(t1_a.grad == 9 * t1_a ** 2)
print(t2_a.grad == -2 * t2_a)


# option 2
t1_b = torch.tensor([2., 3.], requires_grad=True)
t2_b = torch.tensor([6., 4.], requires_grad=True)
cost_b = 3*t1_b**3 - t2_b**2

external_grad = torch.tensor([1., 1.])  # ?????????
cost_b.backward(gradient=external_grad)
print(t1_b.grad == 9 * t1_b ** 2)
print(t2_b.grad == -2 * t2_b)
