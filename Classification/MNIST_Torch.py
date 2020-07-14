# The accuracy goes upto 98% by end of the epochs

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Initializations
kernel_size = 5
learning_rate = 0.01
momentum = 0.5
training_batch_size = 64
testing_batch_size = 1024
shuffle_flag = True
training_losses = []
testing_losses = []
iter_train = []
iter_test = []
epochs = 3
log_base = 10

# Define the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size = kernel_size)
        self.conv2 = nn.Conv2d(10, 20, kernel_size = kernel_size)
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)  # since we have 10 probabilities for 10-digits
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.dropout(self.conv2(x))), 2)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        x = F.log_softmax(x)
        return x

# Initialize network
net = Net()

# Optimizer
optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum = momentum)

# Load training data
training_data = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train = True, download = True,
                              transform = torchvision.transforms.Compose(
                                  [torchvision.transforms.ToTensor(),
                                  torchvision.transforms.Normalize(
                                      (0.1307,), (0.3081,))
                                  ])),
    batch_size = training_batch_size, shuffle = shuffle_flag) 
    

# Load test data
testing_data = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train = False, download = True,
                              transform = torchvision.transforms.Compose(
                                  [torchvision.transforms.ToTensor(),
                                  torchvision.transforms.Normalize(
                                      (0.1307,), (0.3081,))
                                  ])),
    batch_size = testing_batch_size, shuffle = shuffle_flag)
    
# Setup training process
def train(epoch):
    net.train()
    for batch_idx, (train_data, train_target) in enumerate(training_data):
        optimizer.zero_grad()
        output = net(train_data)
        loss = F.nll_loss(output, train_target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_base == 0:
            print('Training Epoch: {}[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(train_data), len(training_data.dataset),
                100.*batch_idx / len(training_data), loss.item()))
            training_losses.append(loss.item())
            iter_train.append((batch_idx * 64) + ((epoch - 1) * len(training_data.dataset)))
            torch.save(net.state_dict(), './model/model.pth')
            torch.save(optimizer.state_dict(), './model/optimizer.pth')

# Setup testing process
iter_test = [i * len(training_data.dataset) for i in range(epochs + 1)]
def test():
    net.eval()
    loss = 0
    accurate = 0
    with torch.no_grad():
        for test_data, test_target in testing_data:
            output = net(test_data)
            loss += F.nll_loss(output, test_target, size_average = False).item()
            prediction = output.data.max(1, keepdim = True)[1]
            accurate += prediction.eq(test_target.data.view_as(prediction)).sum()
    loss /= len(testing_data.dataset)
    testing_losses.append(loss)
    print('\nTest Set: Avg Loss: {:.4f}, Accuracy: {}/{}({:.0f}%)\n'.format(
        loss, accurate, len(testing_data.dataset),
        100.*accurate / len(testing_data.dataset)))

# Initiate training and validation
test()
for epoch in range(1, epochs + 1):
    train(epoch)
    test()

# Visualization
fig = plt.figure()
plt.plot(iter_train, training_losses, color = 'cyan')
plt.scatter(iter_test, testing_losses, color = 'green')
plt.legend(['Training Loss', 'Testing Loss'], loc = 'upper right')
plt.xlabel('Number of training examples seen')
plt.ylabel('Negative log likelihood loss')
fig
