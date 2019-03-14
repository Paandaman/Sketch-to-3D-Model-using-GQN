import torch
from attention import Self_Attention

class Generator(torch.nn.Module):

    def __init__(self):
        stride = 2
        self.gamma = 0
        super(Generator, self).__init__()
        self.linear1 = torch.nn.Linear(100, 1024*4*4)
        self.conv0bn = torch.nn.BatchNorm2d(1024+16)
    
        self.conv1 = torch.nn.utils.spectral_norm(torch.nn.ConvTranspose2d(1024+16, 512, kernel_size = 4, stride=stride, padding=1)) # 8x8
        self.conv1bn = torch.nn.BatchNorm2d(512+7)

        self.conv2 =  torch.nn.utils.spectral_norm(torch.nn.ConvTranspose2d(512+7, 512, kernel_size = 4, stride=stride, padding=1)) # 16x16
        self.conv2bn = torch.nn.BatchNorm2d(512+1)

        self.conv3 = torch.nn.utils.spectral_norm(torch.nn.ConvTranspose2d(512+1, 256, kernel_size = 4, stride=stride, padding=1)) # 32x32
        self.conv3bn = torch.nn.BatchNorm2d(256)

        self.conv4 = torch.nn.utils.spectral_norm(torch.nn.ConvTranspose2d(256, 64, kernel_size = 4, stride=stride, padding=1))  # 64x64
        self.attention4 = Self_Attention(64, int(64/8), 64, self.gamma)
        self.conv4bn = torch.nn.BatchNorm2d(64+7)

        self.conv5 = torch.nn.utils.spectral_norm(torch.nn.ConvTranspose2d(64+7, 3, kernel_size = 3, stride=1, padding=1))   # 64x64
        self.tanh = torch.nn.Tanh()

    def forward(self, x, r, v):

        x = self.linear1(x)

        r_lin = r.view(-1, 1*1*256)
        x = torch.cat([x, r_lin], dim=1)
        x = x.view(-1, 1024+16, 4, 4) 
        x = self.conv0bn(x)
        
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)  
        
        v_1 = v.view(v.size(0), -1, 1, 1) 
        v_1 = v_1.repeat(1, 1, 8, 8) 
        x = torch.cat([x, v_1], dim=1)
        x = self.conv1bn(x)
        
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        
        r = r.view(-1, 1, 16, 16)
        x = torch.cat([x, r], dim=1)
        x = self.conv2bn(x)
        
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3bn(x)
        
        x = self.attention4(self.conv4(x))
        x = torch.nn.functional.relu(x)
        
        v = v.view(v.size(0), -1, 1, 1)
        v = v.repeat(1, 1, 64, 64)
        x = torch.cat([x, v], dim=1)
        x = self.conv4bn(x)
        
        x = self.conv5(x)
        x = self.tanh(x)

        return x
