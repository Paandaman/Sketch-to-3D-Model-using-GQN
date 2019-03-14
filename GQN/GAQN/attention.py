import torch
# Custom NN Modules
class Self_Attention(torch.nn.Module):
    # Calculates the self-attention output for the
    # sent in layer activations 
    # C is channels in feature map. C_ is set to C/2 in the non local neural network paper but C/8 in Sagan, 
    # N is the number of positions in input(to normalize if we have variable input size)
    # gamma is a scale parameter init as zero and gradually increased
    def __init__(self, C, C_, N_1dim, gamma): # Using notation from paper
        super(Self_Attention, self).__init__()
        self.gamma = gamma
        self.C = C
        self.C_ = C_
        self.N_1dim = N_1dim
        self.N = N_1dim*N_1dim
        self.f_x = torch.nn.Conv2d(C, C_, 1, stride = 1)
        self.g_x = torch.nn.Conv2d(C, C_, 1, stride = 1) 
        self.h_x = torch.nn.Conv2d(C, C, 1, stride = 1)
        self.softmax = torch.nn.Softmax(dim=1)

        # Motivation for dimensions
        #input = torch.randn(128, 20) C, N # y = xA^T+b , linuear(in, out)
        #m = torch.nn.Linear(20, 30) C_, C (C x N) ()  W = (Out, IN) --> (128, 20) (30, 20)T --> correct 
        #output = m(input)       (C x N)T (C_ x C)T --> N x C_ T -- > C_ x N

    def forward(self, x_activation):

        f_x = self.f_x(x_activation)
        f_x = torch.transpose(f_x.view(-1, self.C_, self.N), 1, 2) # transpose dim 1 and 2 but keep batch in order
        g_x = self.g_x(x_activation) 
        g_x = g_x.view(-1, self.C_, self.N) # f_x is [-1, N, C_] , g_x --> [-1, C_, N]
        S = torch.bmm(f_x, g_x) # [-1, N, N]
        B = self.softmax(S) # [-1, N, N], rescale each row
        h_x = self.h_x(x_activation)
        h_x = h_x.view(-1, self.C, self.N) # [-1, C, N]
        h_x = torch.transpose(h_x, 1, 2)
        O = torch.bmm(B,h_x)  # [-1, N, C] Self-attention feature maps
        O = torch.transpose(O, 1, 2) # [-1, C, N]
        O = O.view(-1, self.C, self.N_1dim, self.N_1dim) # Reshape into original feature map shape, [-1, C, width, height]
        y = torch.add(self.gamma*O, x_activation) # [-1, C, width, height]
        return y