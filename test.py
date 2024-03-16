
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
logits = torch.Tensor([[0, 0, 0, 0, 1, 1, 1, 0, 0 ,0], [0, 0, 0, 0, 1, 1, 1, 0, 0 ,0]])
labels = torch.Tensor([5, 3])
radius = 1

labels = labels.long()
labels = nn.functional.one_hot(labels, num_classes=10)
for label in labels:
    idx = torch.argmax(label)
    for i in range(idx - int(radius), idx + int(radius) + 1):
        if i >= 0 and i < len(label):
            label[i] = 1

labels = labels / labels.sum(dim = 1, keepdim=True)

loss = criterion(logits, labels)
print(loss.item())
            