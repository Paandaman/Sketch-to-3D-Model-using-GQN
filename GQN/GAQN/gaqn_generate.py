from arrange_data import arrange_data_generation
from torchvision.utils import make_grid
from representator import Representator
from generator import Generator
from torch.utils import data
from gqn_dataset import *
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch

def convert_img_for_display(gen_x):
    OldMin = -1
    OldMax = 1
    NewMin = 0
    NewMax = 1
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    img = ((gen_x - OldMin)*NewRange/OldRange+NewMin)
    return img

def generate_image(M, x_real, v_real, query_view, skip_indx, latent_sample):

    skip_indx = skip_indx.data[0]
    x_real = x_real.float()
    x_real = x_real.cuda()
    v_real = v_real.cuda()
    query_view = query_view.cuda()

    zeros = torch.zeros(1,1).cuda() 
    latent_size = 100
    loss = 0

    # generate samples
    x_lat = latent_sample.view(-1, 100).cuda() 
    r = torch.zeros(1, 256, 1, 1).cuda()

    for view in range(0, M):
        if view != skip_indx:
            x_s = x_real[view, :, :, :]
            v_s = v_real[view, :]
            r_s = rep(x_s.unsqueeze(0), v_s.unsqueeze(0))
            r += r_s
        elif view == skip_indx:
            continue

    r = torch.div(r, M)

    gen_x = gen(x_lat, r, query_view)
    return gen_x

def get_data(randn):
    i = 0
    for t, data in enumerate(trainloader, 0):
        for scene in data:
            i += 1
            if i == randn:
                data_tmp = scene
                return data_tmp

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

def main():
    x = get_data(1)
    M = 15
    x_real, v_real, query_view, skip_indx, critic_img = arrange_data_generation(x, M)

    imglist = []
    imglist_real = []
    i = 0
    latent_size = 100
    latent_sample = latent_distr.sample(torch.Size([latent_size, 1])).view(1, latent_size)

    for view in v_real: 
        skip_indx = torch.LongTensor([i])
        gen_img = generate_image(M, x_real, v_real, view.unsqueeze(0), skip_indx, latent_sample)
        real_view = x_real[i,:,:,:]
        gen_x = convert_img_for_display(gen_img.cpu().detach().squeeze()) 
        real_xx = convert_img_for_display(real_view.cpu())
        imglist.append(gen_x)
        imglist_real.append(real_xx)
        i += 1

    plt.figure() 
    show(make_grid(imglist))
    plt.figure()
    show(make_grid(imglist_real))
    plt.show()


parser = argparse.ArgumentParser(description='Self-Attention Generative Adversarial Networks for GQN')
parser.add_argument('--representation_setup', type=str, default='separate', help='Use shared or separate representation network')
parser.add_argument('--batch_size', type=int, default='1', help='Batch size')
parser.add_argument('--test_dataset', type=str, default='.', help='Path to test data')
parser.add_argument('--path_to_model', type=str, default='.', help='Path to trained model')

opt = parser.parse_args()

torch.cuda.set_device(0)

path = opt.path_to_model
latent_distr = torch.distributions.normal.Normal(0, 1)


if opt.representation_setup == "same":
    print("Same Representation Networks")
    gen = Generator()
    rep = Representator()
    gen.cuda()
    rep.cuda()

    checkpoint = torch.load(path)
    gen.load_state_dict(checkpoint['Gen_state_dict'])
    rep.load_state_dict(checkpoint['Rep_state_dict'])

    gen.eval()
    rep.eval()

    gen.cuda()
    rep.cuda()

elif opt.representation_setup == "separate":
    print("Separate Representation Networks")
    gen = Generator()
    rep = Representator()

    checkpoint = torch.load(path)
    gen.load_state_dict(checkpoint['Gen_state_dict'])
    rep.load_state_dict(checkpoint['Rep_gen_state_dict'])

    gen.eval()
    rep.eval()

    gen.cuda()
    rep.cuda()


tf.enable_eager_execution()
path = opt.test_dataset 
train_data = GqnDataset(path)
trainloader = data.DataLoader(train_data)

main()

