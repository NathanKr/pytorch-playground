import torchvision
import torch

# this is the neural network
model = torchvision.models.resnet18(pretrained=True)
print(f"torchvision.models.resnet18(pretrained=True)\n{model}")

data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# forward pass
prediction = model(data)

loss = (prediction - labels).sum()

# backward pass
loss.backward()

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# gradient descent
optim.step()
