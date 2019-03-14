import torch
from attention import Self_Attention

class Critic(torch.nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.gamma = 0
        v_dim = 7
        self.fin_size = 3
        self.concat_size = 64+7+1
        self.conv1 = torch.nn.utils.spectral_norm(torch.nn.Conv2d(3, 64, 4, stride=2, padding=1))  # 32
        self.conv1bn = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.utils.spectral_norm(torch.nn.Conv2d(64, 64, 4, stride=2, padding=1)) # 16
        self.conv2bn = torch.nn.BatchNorm2d(self.concat_size)
        self.conv3 = torch.nn.utils.spectral_norm(torch.nn.Conv2d(self.concat_size, self.concat_size, 5, stride=2)) # 6
        self.conv3bn = torch.nn.BatchNorm2d(self.concat_size)
        self.conv4 = torch.nn.utils.spectral_norm(torch.nn.Conv2d(self.concat_size, 512, 2, stride=2)) # 3x3
        self.attention4 = Self_Attention(512, int(512/8), 3, self.gamma)
        self.conv4bn = torch.nn.BatchNorm2d(512+7)
        self.conv5 = torch.nn.utils.spectral_norm(torch.nn.Conv2d(512+7, 512, 1, stride=1))
        self.fc1 = torch.nn.utils.spectral_norm(torch.nn.Linear(512*self.fin_size*self.fin_size, 1))

    def forward(self, x, v, r):

        x = torch.nn.functional.leaky_relu(self.conv1(x))
        # Broadcast V to same size as hidden layer
        v_1 = v.view(v.size(0), -1, 1, 1)
        v_1 = v_1.repeat(1, 1, 16, 16)

        x = self.conv1bn(x)
        x = torch.nn.functional.leaky_relu(self.conv2(x))
        
        r = r.view(-1, 1, 16, 16)
         # Move channels to the same dimension as x and v
        x = torch.cat([x, r], dim=1) 
        x = torch.cat([x, v_1], dim=1)
        x = self.conv2bn(x)
        
        x = self.conv3(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv3bn(x)
        x = torch.nn.functional.leaky_relu(self.conv4(x))
        
        v = v.view(v.size(0), -1, 1, 1)
        v = v.repeat(1, 1, 3, 3)
        x = torch.cat([x, v], dim=1)
        x = self.conv4bn(x)
        
        x = torch.nn.functional.leaky_relu(self.conv5(x))
        x = x.view(-1, 512*self.fin_size*self.fin_size)
        x = self.fc1(x)

        return x
