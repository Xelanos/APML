import torch
from dataset import *
from models import SimpleModel
from transforms import AddGaussianNoise


datasets = [MyDataset(get_dataset_as_array())]

for _ in range(3):
    noisey_dataset = MyDataset(get_dataset_as_array(), AddGaussianNoise())
    datasets.append(noisey_dataset)


dataset = torch.utils.data.ConcatDataset(datasets)

datasetloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)

model = SimpleModel()
checkpoint = torch.load('./data/pre_trained.ckpt')
model.load_state_dict(checkpoint['model_state_dict'])

correct_test = 0
total = 0
with torch.no_grad():
    for data in datasetloader:
        images, labels = data
        outputs = model(images)
        predicted = torch.argmax(outputs.data, dim=1)
        total += labels.size(0)
        correct_test += (predicted == labels).sum().item()
print(f'total: {total} ')
print(f'Model Accruacy is : {100 * correct_test / total:.2f}%')


