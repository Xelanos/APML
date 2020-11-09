from dataset import *
from models import SimpleModel
import torch
from torch.utils.data import RandomSampler

EPOCHS = 120

labels = label_names()
dataset = get_dataset_as_torch_dataset()

trainset, testset = torch.utils.data.random_split(dataset, [len(dataset) - 500, 500])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True)
testloader = torch.utils.data.DataLoader(trainset, batch_size=500, shuffle=True)

classes = tuple(labels.values())

model = SimpleModel()
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1e-4
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-2, lr_decay=1e-5)


for epoch in range(EPOCHS):  # loop over the dataset multiple times
    print(f"EPOCH {epoch + 1}/{EPOCHS}", end='')

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

    correct_test = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            predicted = torch.argmax(outputs.data, dim=1)
            total += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    print(f'Average loss is {running_loss / len(trainset):.4f}')
    print(f'Train accruacy is : {100 * correct_train / len(trainset):.2f}%')
    print(f'Test accruacy is : {100 * correct_test / total:.2f}%')
    print("")

    if epoch == EPOCHS - 2:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, "data/last_step.ckpt")



print('Finished Training')

torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, "data/my_model.ckpt")






