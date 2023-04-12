import torch
import torch.nn as nn
import torch.nn.functional as F

model = 'checkpoints/best_model.pth'
input = torch.randn(32, 1024, 3)
#output, max_indices1, max_indices

class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)
        max_indices = x[1] # get the indices of the max values
        x = x[0]
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        iden = torch.eye(3, requires_grad=True).repeat(batch_size,1,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x, max_indices


class PointNet(nn.Module):
    def __init__(self, num_classes=40):
        super(PointNet, self).__init__()
        self.tnet1 = TNet(k=3)
        self.tnet2 = TNet(k=64)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.transpose(2, 1)
        t, max_indices1 = self.tnet1(x)
        x = torch.bmm(t, x)
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        t, max_indices2 = self.tnet2(x)
        x = torch.bmm(t, x)
        x = x.transpose(2, 1)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)
        max_indices3 = x[1] # get the indices of the max values
        x = x[0]
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        #print(max_indices3)
        return x, max_indices1, max_indices2, max_indices3