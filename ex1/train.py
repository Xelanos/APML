from dataset import *
from models import SimpleModel
import torch
from torch.utils.data import RandomSampler


labels = label_names()
dataset = get_dataset_as_torch_dataset()

trainset, testset = torch.utils.data.random_split(dataset, [len(dataset) - 2000, 2000])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True)
testloader = torch.utils.data.DataLoader(trainset, batch_size=500, shuffle=True)

classes = tuple(labels.values())

model = SimpleModel()
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1e-4
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-2, lr_decay=1e-5)


for epoch in range(500):  # loop over the dataset multiple times
    print(f"EPOCH {epoch}", end='')

    running_loss = 0.0
    correct_train = 0
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

        # collect statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct_train += (predicted == labels).sum().item()

    print(" - done")
    print(f'Average loss is {running_loss / len(trainset):.4f}')
    print(f'Train accruacy is : {100 * correct_train / len(trainset):.2f}%')

    correct_test = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    print(f'Test accruacy is : {100 * correct_test / total:.2f}%')
    print("")

print('Finished Training')






