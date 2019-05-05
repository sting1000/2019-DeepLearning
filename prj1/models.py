import os

# third-party library
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(5),
    transforms.ToTensor()
])

class rotated_dataset():
    def __init__(self, data, transform=transform):
        self.data = data
        self.transform = transform

    def __getitem__(self,idx1,idx2):
        item = self.data[idx1][idx2].unsqueeze(0)
        item = self.transform(item)
        return item.squeeze()
    
    def __rotate__(self):
        trans_img = torch.zeros_like(self.data)
        for i in range(trans_img.shape[0]):
            for j in range(trans_img.shape[1]):
                trans_img[i][j] = self.__getitem__(i,j)
        return trans_img
    
def compute_nb_errors(model, input, target):
    nb_errors = []
    num_round = 20
    mini_batch_size = input.size(0)//num_round
    for i in range(num_round):
        b = i*mini_batch_size
        output = model(input.narrow(0, b, mini_batch_size))[0]
        target_patch = target.narrow(0, b, mini_batch_size)
        _,output = output.max(1)
        nb_errors.append((output != target_patch).sum().data.item()/mini_batch_size)
    nb_tensor = torch.FloatTensor(nb_errors)
    return nb_tensor.mean().item(),nb_tensor.std().item()


class baseline(nn.Module):
    def __init__(self):
        super(baseline, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (2, 14, 14)
            nn.Conv2d(
                in_channels=2,              # input height
                out_channels=16,            # n_filters
                kernel_size=2,              # filter size
                stride=1                   # filter movement/step
            ),                              
#             nn.Dropout(0.5),
            nn.ReLU(),                      # activation
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),    # choose max value in 2x2 area
        )
        self.conv2 = nn.Sequential(        
            nn.Conv2d(16, 64, 2, 1, 1),     
#             nn.Dropout(0.5),
            nn.ReLU(),                      # activation
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),                
        )
        self.conv3 = nn.Sequential(        
            nn.Conv2d(64, 128, 2, 1, 1),     
#             nn.Dropout(0.5),
            nn.BatchNorm2d(128),
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                
        )
        self.fc1 = nn.Linear(128 *2 * 2, 64)
        self.fc2 = nn.Linear(64, 20)
        self.out = nn.Linear(20,2)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.out(x)
        return output,x

class CNN_Siamese(nn.Module):
    def __init__(self):
        super(CNN_Siamese, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 14, 14)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=2,              # filter size
                stride=1                   # filter movement/step
            ),                              
#             nn.Dropout(0.5),
            nn.ReLU(),                      # activation
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 64, 2, 1, 1),     
#             nn.Dropout(0.5),
            nn.ReLU(),                      # activation
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),                
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(64, 128, 2, 1, 1),     
#             nn.Dropout(0.5),
            nn.BatchNorm2d(128),
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                
        )
        self.fc1 = nn.Linear(128 *2 * 2, 64)
        self.fc2 = nn.Linear(64, 10)
        self.out = nn.Linear(20,2)

    def forward_once(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        output = self.fc2(x)
        return output,x
    
    def forward(self,x):
        input1 = x.narrow(1,0,1)
        input2 = x.narrow(1,1,1)
        output1,last_layer1 = self.forward_once(input1)
        output2,last_layer2 = self.forward_once(input2)
        output = torch.cat([output1,output2],dim=1)
        output = self.out(output).squeeze()
        return output,last_layer1,last_layer2,output1,output2

    
def ModelTest(model_,train_set,test_set,auxiliary_loss,mini_batch_size=100,EPOCH=25,LR=0.001):
    
    model = model_
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all cnn parameters
    criterion = nn.CrossEntropyLoss()                       # the target label is not one-hotted

    train_input = train_set['input']
    train_classes = train_set['classes']
    train_target = train_set['target']
    
    test_input = test_set['input']
    test_classes = test_set['classes']
    test_target = test_set['target']
    
    for epoch in range(EPOCH):
        for b in range(0, train_input.size(0), mini_batch_size):   # gives batch data, normalize x when iterate train_loader

            x_tr = train_input.narrow(0, b, mini_batch_size)
            y_tr_class = train_classes.narrow(0, b, mini_batch_size)
            y_tr_target = train_target.narrow(0, b, mini_batch_size)#.float()
            x_tr = Variable(x_tr)   # batch x
            y_tr_class = Variable(y_tr_class)   # batch y
            
            if auxiliary_loss:
                if model_.__class__.__name__ == 'baseline':
                    output,last_layer1 = model(x_tr)[0:2]
                    loss = criterion(last_layer1, y_tr_class.narrow(1,0,1).squeeze()) + criterion(output, y_tr_target)
                else:
                    output,last_layer1,last_layer2,output1,output2 = model(x_tr)               # cnn output
                    auxiliary_loss = criterion(output1, y_tr_class.narrow(1,0,1).squeeze())+criterion(output2, y_tr_class.narrow(1,1,1).squeeze())
                    loss = auxiliary_loss + 0.3*criterion(output,y_tr_target)   # cross entropy loss
            else:
                output,last_layer1 = model(x_tr)[0:2]               # cnn output
                loss = criterion(output, y_tr_target)   # cross entropy loss                
            
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

        train_error_mean, train_error_std = compute_nb_errors(model, train_input, train_target)
        test_error_mean, test_error_std = compute_nb_errors(model, test_input, test_target)

        print('Epoch: ', epoch, '| test error: %.2f' % test_error_mean,'|standard deviation: %.2f'%test_error_std,'\n\n')
    
    return model, last_layer1