import torch

grads = {}

a = torch.tensor([1.0,2.0,3.0],requires_grad=True)
loss = torch.tensor([2.0,3.0,10.0],requires_grad=True)
# b = (a-3).exp()/((a-3).exp().sum())
c = a.softmax(0)
c.backward(loss)
d = a.grad


# c.backward(a)
print(a.grad)
# print(b)
print(c)
print(d)
grad = c.unsqueeze(1) * c.unsqueeze(0)
for i in range(len(grad)):
    grad[i, i] -= c[i]
grad = - grad

print(grad)
print((grad*loss).sum(1))
