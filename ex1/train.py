from dataset import *
from models import SimpleModel
import torch
from torch.utils.data import RandomSampler


labels = label_names()
trainset = get_dataset_as_array()

trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True)

classes = tuple(labels.values())

model = SimpleModel()
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1e-4
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-2, lr_decay=1e-5)


for epoch in range(500):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')






