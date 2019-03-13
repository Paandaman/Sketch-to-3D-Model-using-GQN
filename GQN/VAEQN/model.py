import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from representation import Pyramid, Tower, Pool
from core import InferenceCore, GenerationCore
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

class GQN(nn.Module):
    def __init__(self, representation="pool", L=12, shared_core=False):
        super(GQN, self).__init__()
        
        # Number of generative layers
        self.L = L
                
        # Representation network
        self.representation = representation
        if representation=="pyramid":
            self.phi = Pyramid()
        elif representation=="tower":
            self.phi = Tower()
        elif representation=="pool":
            self.phi = Pool()
            
        # Generation network
        self.shared_core = shared_core
        if shared_core:
            self.inference_core = InferenceCore()
            self.generation_core = GenerationCore()
        else:
            self.inference_core = nn.ModuleList([InferenceCore() for _ in range(L)])
            self.generation_core = nn.ModuleList([GenerationCore() for _ in range(L)])
            
        self.eta_pi = nn.Conv2d(128, 2*1, kernel_size=5, stride=1, padding=2)
        self.eta_g = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)
        self.eta_e = nn.Conv2d(128, 2*1, kernel_size=5, stride=1, padding=2)

    # EstimateELBO
    def forward(self, x, v, v_q, x_q, sigma, skip_indx):
        B, M, *_ = x.size()
        
        # Scene encoder - sketch images are used here
        if self.representation=='tower':
            r = x.new_zeros((B, 256, 16, 16))
        else:
            r = x.new_zeros((B, 256, 1, 1))
        
        # Don't encode query image
        for k in range(M):
            if k != skip_indx:
                r_k = self.phi(x[:, k], v[:, k])
                r += r_k

        # From here on real images are used

        # Generator initial state
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))

        # Inference initial state
        c_e = x.new_zeros((B, 128, 16, 16))
        h_e = x.new_zeros((B, 128, 16, 16))
                
        elbo = 0
        for l in range(self.L):
            # Prior factor
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_g), 1, dim=1)
            std_pi = torch.exp(0.5*logvar_pi)
            pi = Normal(mu_pi, std_pi)
            
            # Inference state update
            if self.shared_core:                
                # x_q here is from real data
                c_e, h_e = self.inference_core(x_q, v_q, r, c_e, h_e, h_g, u)
            else:
                c_e, h_e = self.inference_core[l](x_q, v_q, r, c_e, h_e, h_g, u)
            
            # Posterior factor
            mu_q, logvar_q = torch.split(self.eta_e(h_e), 1, dim=1)
            std_q = torch.exp(0.5*logvar_q)
            q = Normal(mu_q, std_q)
            
            # Posterior sample
            z = q.rsample()
            
            # Generator state update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)
                
            # ELBO KL contribution update
            elbo -= torch.sum(kl_divergence(q, pi), dim=[1,2,3])
                
        # ELBO likelihood contribution update
        elbo += torch.sum(Normal(self.eta_g(u), sigma).log_prob(x_q), dim=[1,2,3])

        return elbo
    
    def generate(self, x, v, v_q, skip_index, i):
        B, M, *_ = x.size()
        
        # Scene encoder
        if self.representation=='tower':
            r = x.new_zeros((B, 256, 16, 16))
        else:
            r = x.new_zeros((B, 256, 1, 1))

        # i = 0 used when generating images
        # after observing zero sketches
        # During training i is set to 99
        if i != 0:
            for k in range(M):
                if k != skip_index:
                    r_k = self.phi(x[:, k], v[:, k])
                    r += r_k

        # Initial state
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))
        
        imglist = []

        for l in range(self.L):
            # Prior factor
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_g), 1, dim=1)
            std_pi = torch.exp(0.5*logvar_pi)# *0. # set to zero to get the maximum a posteriori estimation
            pi = Normal(mu_pi, std_pi)
            
            # Prior sample
            z = pi.sample()
            
            # State update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)
            
            ## Uncomment all below to save image per roll out
            #imglist.append(torch.clamp(self.eta_g(u), 0, 1).squeeze(0).cpu().detach())

        ## Save image per roll out        
        """
        plt.figure()#
        self.show(make_grid(imglist, nrow=self.L))
        # Remove metrics on axes
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('L')
        plt.show()
        """
        ###


        # Image sample
        mu = self.eta_g(u)

        return torch.clamp(mu, 0, 1)
    
    def kl_divergence(self, x, v, v_q, x_q, skip_indx, i):
        B, M, *_ = x.size()

        # Scene encoder
        if self.representation=='tower':
            r = x.new_zeros((B, 256, 16, 16))
        else:
            r = x.new_zeros((B, 256, 1, 1))
        # i = 0 used when calculating Bayesian Surprise
        # to measure KL divergence after observing zero sketches
        # During training i is set to 99
        if i != 0: 
            for k in range(M):
                if k != skip_indx:
                    r_k = self.phi(x[:, k], v[:, k])
                    r += r_k

        # Generator initial state
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))

        # Inference initial state
        c_e = x.new_zeros((B, 128, 16, 16))
        h_e = x.new_zeros((B, 128, 16, 16))
                
        kl = 0
        for l in range(self.L):
            # Prior factor
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_g), 1, dim=1)
            std_pi = torch.exp(0.5*logvar_pi)
            pi = Normal(mu_pi, std_pi)
            
            # Inference state update
            if self.shared_core:
                c_e, h_e = self.inference_core(x_q, v_q, r, c_e, h_e, h_g, u)
            else:
                c_e, h_e = self.inference_core[l](x_q, v_q, r, c_e, h_e, h_g, u)
            
            # Posterior factor
            mu_q, logvar_q = torch.split(self.eta_e(h_e), 1, dim=1)
            std_q = torch.exp(0.5*logvar_q)
            q = Normal(mu_q, std_q)

            # Posterior sample
            z = q.rsample()
            
            # Generator state update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)
                
            # ELBO KL contribution update
            kl += torch.sum(kl_divergence(q, pi), dim=[1,2,3])

        return kl
    
    def reconstruct(self, x, v, v_q, x_q, skip_indx):
        B, M, *_ = x.size()

        # Scene encoder
        if self.representation=='tower':
            r = x.new_zeros((B, 256, 16, 16))
        else:
            r = x.new_zeros((B, 256, 1, 1))
        for k in range(M):
            if k != skip_indx:
                r_k = self.phi(x[:, k], v[:, k])
                r += r_k
            
        # Generator initial state
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))

        # Inference initial state
        c_e = x.new_zeros((B, 128, 16, 16))
        h_e = x.new_zeros((B, 128, 16, 16))
                
        for l in range(self.L):
            # Inference state update
            if self.shared_core:
                c_e, h_e = self.inference_core(x_q, v_q, r, c_e, h_e, h_g, u)
            else:
                c_e, h_e = self.inference_core[l](x_q, v_q, r, c_e, h_e, h_g, u)
            
            # Posterior factor
            mu_q, logvar_q = torch.split(self.eta_e(h_e), 1, dim=1)
            std_q = torch.exp(0.5*logvar_q)
            q = Normal(mu_q, std_q)
            
            # Posterior sample
            z = q.rsample()
            
            # Generator state update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)
                
        mu = self.eta_g(u)

        return torch.clamp(mu, 0, 1)
    
    def show(self, img):
        # Used for saving roll out image
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)))