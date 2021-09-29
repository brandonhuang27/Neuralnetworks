import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784 # 28x28 input neurons
hidden_size = 500 # 500 neurons in hidden layers
num_classes = 10 # 10 different classes (0-9)
num_epochs = 5 # 5 iterations
batch_size = 100 # Trains 100 neurons in each batch
learning_rate = 0.001 # Controls how much to adjust weights according to the error/loss after each iteration

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True) # downloads training data

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor()) # downloads testing data

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) # creates a linear layer - input size is the input size, output size is the hidden size
        self.relu = nn.ReLU() # initializes the rectified linear unit (relu) activation function
        self.fc2 = nn.Linear(hidden_size, num_classes) # creates another linear layer - input size is the hidden size, output size is the number of classes
    
    def forward(self, x): # takes in a sample x
        out = self.fc1(x) # applies the first linear function
        out = self.relu(out) # applies the rectified linear unit activation function
        out = self.fc2(out) # applies the second linear function
        return out # returns the output

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() # applies the cross-entropy loss function (which applies the softmax function) to calculate error
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # applies the parameters and learning rate to create optimizer 

# Train the model
total_step = len(train_loader) # the total number of steps to train the model is the length of the training loader
for epoch in range(num_epochs): # loop over the epochs
    for i, (images, labels) in enumerate(train_loader): # loop over all the batches
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(device) # reshapes the input tensor and pushes it to the device
        labels = labels.to(device) # pushes the labels to the device
        
        # Forward pass
        outputs = model(images) # inputs the image into the model and stores the output
        loss = criterion(outputs, labels) # calculates loss using the predicted outputs and actual labels
        
        # Backward and optimize
        optimizer.zero_grad() # empties the values in the gradients
        loss.backward() # backpropagation
        optimizer.step() # updates the parameters
        
        if (i+1) % 100 == 0: # every 100 iterations
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item())) # print the current epoch, the number of epochs, the current step, the total number of steps, and the loss

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0 # number of correct predictions
    total = 0 # number of total samples
    for images, labels in test_loader: # loop over all of the batches in the test samples
        images = images.reshape(-1, 28*28).to(device) # reshapes the input tensor and pushes it to the device
        labels = labels.to(device) # pushes the labels to the device
        outputs = model(images) # inputs the test images into the model and stores the outputs
        _, predicted = torch.max(outputs.data, 1) # returns the value and the index
        total += labels.size(0) # increase by the number of samples in the current batch
        correct += (predicted == labels).sum().item() # update the number of correct predictions

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total)) # print the accuracy

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
