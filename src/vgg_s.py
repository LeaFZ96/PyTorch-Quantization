import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=400,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=80,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = models.vgg11()



print('device: ', device)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0025, momentum=0.8)

start_time = time.time()
last_time = start_time
print("start train")

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    read_time = 0.0
    forward_time = 0.0
    backward_time = 0.0
    optimize_time = 0.0
    time_0 = time.time()
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].cuda(), data[1].cuda()

        # zero the parameter gradients
        time_1 = time.time()
        optimizer.zero_grad()


        # forward + backward + optimize
        outputs = net(inputs)

        time_2 = time.time()
        loss = criterion(outputs, labels)
        loss.backward()

        time_3 = time.time()
        optimizer.step()

        time_4 = time.time()

        read_time += time_1 - time_0
        forward_time += time_2 - time_1
        backward_time += time_3 - time_2
        optimize_time += time_4 - time_3

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 2000 mini-batches
            time_5 = time.time()
            print('epoch: %d\t iter: %5d\t loss: %.3f\t total_t: %.3f\t read_t: %.4f\t forward_t: %.4f\t backward_t: %.4f\t optimize_t: %.4f\t' %
                  (epoch + 1, i + 1, running_loss / 20, time.time() - last_time, read_time, forward_time, backward_time, optimize_time))
            time_6 = time.time()
            last_time = time.time()
            running_loss = 0.0
            read_time = 0.0
            forward_time = 0.0
            backward_time = 0.0
            optimize_time = 0.0
        time_0 = time.time()

print('total time: %.3f' % (time.time() - start_time))
print('Finished Training')

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


total = 0
correct = 0
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
    total += class_total[i]
    correct += class_correct[i]
    

print('Total accuracy: %2d %%' % (100 * correct / total))