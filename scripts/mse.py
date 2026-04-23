import torch

from computational_graph import CompGraph, ValueNode
from computational_graph.nodes.loss.mse import MSENode

# Own implementation
x1 = ValueNode("x1")
x2 = ValueNode("x2")
loss = ValueNode("loss")

mse = MSENode(x1, x2, loss)

cg = CompGraph([x1, x2], [loss])

cg.forward([2.0, 8.0])
cg.backward()

print("Own Implementation:")
print(f"loss: {loss.v}")
print(f"y_pred grad: {x1.grad_v}")
print("")

# PyTorch
x1_torch = torch.tensor([2.0], requires_grad=True)
loss_torch = torch.nn.functional.mse_loss(x1_torch, torch.tensor([8.0]))
loss_torch.backward()

print("PyTorch:")
print(f"loss: {loss_torch}")
print(f"y_pred grad: {x1_torch.grad}")
