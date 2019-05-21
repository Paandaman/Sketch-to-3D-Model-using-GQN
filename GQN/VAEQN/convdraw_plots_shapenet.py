import torch
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from matplotlib import animation
from model import GQN
from skimage import io, transform
from torch.utils import data
from arrange_data import arrange_data
from camera import Camera_view        
from bayesian_surprise import bayesian_surprise
from perceptual_loss import perceptual_loss
from load_data import get_data
from shape_net_dataset_sketch import ShapeNetDatasetSketch

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

def get_viewpoint(phi, theta, v_real):
    # Calculates the view point from given phi and theta angles
    # and class id from a view point from test dataset
    r = 2
    x = r*np.sin(phi[0])*np.cos(theta[0])
    y = r*np.sin(phi[0])*np.sin(theta[0])
    z = r*np.cos(theta[0])
    p1 = np.cos(theta[0]) 
    p2 = np.sin(theta[0])
    p3 = np.cos(phi[0])
    p4 = np.sin(phi[0])
    class_id = v_real[0][0][-1]
    return [x, y, z, p1, p2, p3, p4, class_id]

def prepare_sketch(img_path):
    # Load and process real sketch
    image = io.imread(img_path)
    image = transform.resize(image, (64, 64))
    image = np.array(image)
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor)
    image = image.permute(0, 1, 4, 2, 3).cuda()

    return image.narrow(dim=2, start=0, length=1)

def main():

    def init():
        im.set_data(np.zeros((64, 64)))

    def animate(i):
        view = block_rot.get_view()
        # Set skip index to anything > 15 to make sure nothing is skipped
        x_mu = model.generate(x_real, v_real, view, 44, 99)  
        data = x_mu.squeeze(0) 
        data = data.repeat(3,1,1)
        data = data.permute(1, 2, 0)
        data = data.cpu().detach()
        im.set_data(data)
        return im

    x = get_data(test_loader,22)
    M = 15
    sigma = 0.7

    print("Set variance of prior to zero when plotting!")

    data_tmp_sketch = x[0]
    data_tmp_real = x[1]

    x_real, v_real, query_view, skip_indx, critic_img = arrange_data(data_tmp_sketch, M, True, 0)
    x_real_real, v_real_real, query_view_real, skip_indx, critic_img_real = arrange_data(data_tmp_real, M, False, skip_indx)

    skip_indx = skip_indx.data[0]
    x_real = x_real.float()
    x_real = x_real.cuda()
    v_real = v_real.cuda()
    query_view = query_view.cuda()
    skip_indx = skip_indx
    critic_img = critic_img.cuda()

    imglist = []
    imglist_sketch = []
    imglist_real = []
    i = 0
    if opt.mode == 'single':
        # Generate image from new view point after having 
        # observed all other view points.
        for s in range(M): 
            skip_indx = torch.LongTensor([i])
            real_view = x_real_real[:,skip_indx,:,:] 
            sketch_ver = x_real[:,skip_indx,:,:]
            real_ver = x_real_real[:,skip_indx,:,:]
            view = v_real[:, skip_indx, :]
            
            x_mu = model.generate(x_real, v_real, view, skip_indx, 99) 
            sketch_xx = sketch_ver.squeeze(0) 
            real_xx = real_ver.squeeze(0)

            imglist.append(x_mu.squeeze(0).cpu().detach())
            imglist_sketch.append(sketch_xx.squeeze(0).cpu().detach())
            imglist_real.append(real_xx.squeeze(0).cpu().detach())
            i += 1

        plt.figure().suptitle('Generated', fontsize=20)
        show(make_grid(imglist))
        plt.figure().suptitle('Sketch for Representation', fontsize=20)
        show(make_grid(imglist_sketch))
        plt.figure().suptitle('Ground Truth', fontsize=20)
        show(make_grid(imglist_real))
        plt.show()

    if opt.mode == 'single_sketch':
        # Generate images from a single sketch
        view_idx = 0
        for s in range(14):
            skip_indx = torch.LongTensor([s+1])
            real_view = x_real_real[:,skip_indx,:,:] 
            sketch_ver = x_real[:,skip_indx,:,:]
            real_ver = x_real_real[:,skip_indx,:,:]
            view = v_real[:, skip_indx, :]

            x_mu = model.generate(x_real[:,view_idx,:,:].unsqueeze(1), v_real[:, view_idx, :].unsqueeze(1), view, skip_indx, 99)
            sketch_xx = sketch_ver.squeeze(0) 
            real_xx = real_ver.squeeze(0)

            imglist.append(x_mu.squeeze(0).cpu().detach())
            imglist_real.append(real_xx.squeeze(0).cpu().detach())
            i += 1

        imglist_sketch.append(x_real[:,view_idx,:,:].squeeze(0).cpu().detach())
        plt.figure().suptitle('Generated', fontsize=20) 
        plt.xticks([])
        plt.yticks([])
        show(make_grid(imglist))
        plt.figure().suptitle('Sketch for Representation', fontsize=20)
        plt.xticks([])
        plt.yticks([])
        show(make_grid(imglist_sketch))
        plt.figure().suptitle('Ground Truth', fontsize=20)
        show(make_grid(imglist_real))
        plt.xticks([])
        plt.yticks([])
        plt.show()

    if opt.mode == 'rotation':
        # Perform mental rotation of a model
        block_rot = Camera_view(v_real[0][0])
        fig = plt.figure()
        data = np.zeros((64, 64))
        im = plt.imshow(data)
        plt.xticks([])
        plt.yticks([])
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=np.arange(0, 200), interval=100)
        #anim.save('/path/m.gif', dpi=80, writer='imagemagick')
        plt.show()

    if opt.mode == 'mental_rotation_plot':
        # A plot version of the mental rotation where 
        # the model is rendered from various view points
        phi = torch.arange(2*np.pi/11, 2*np.pi, 2*np.pi/11)
        theta = torch.arange(np.pi/11, np.pi, np.pi/11)
    
        # First viewpoint, initialize model position
        v = get_viewpoint(phi, theta, v_real)        
  
        block_rot = Camera_view(v)
        # Generate images
        imglist = []
        for theta_angle in theta:
            for phi_angle in phi:
                view = block_rot.plot_grid(theta_angle, phi_angle)
                x_mu = model.generate(x_real, v_real, view, 44, 99) 
                imglist.append(x_mu.squeeze(0).cpu().detach())
                
        plt.figure().suptitle('Predictions', fontsize=10)
        show(make_grid(imglist, nrow=len(phi)))
        plt.xlabel('Yaw')
        plt.ylabel('Pitch')
        # Remove metrics on axes
        plt.xticks([])
        plt.yticks([])
        plt.show()
        #plt.savefig('phithetaplot.png')

    if opt.mode == 'real_sketch_rotation':
        # Mental rotation of a real sketch 
        block_rot = Camera_view(v_real[0][1])

        x_real = prepare_sketch(opt.path_to_real_sketch)
        # Need to specify a view point. Code below extracts 
        # one from a random view point in the test data set
        v_real = v_real.unsqueeze(0)[:,:,4,:] 

        fig = plt.figure()
        data = np.zeros((64, 64))
        im = plt.imshow(data)
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=np.arange(0, 200), interval=200)
        #anim.save('/save/model.gif', dpi=80, writer='imagemagick')
        plt.show()        
    

    if opt.mode == 'real_sketch_single_image':
        # Generate images from a single, real sketch
        # Need to provide information of what class is in the sketch
        # Here it is obtained from the test dataset from a model from the same class
        v_sketch = v_real[0][3] 
        x_real = prepare_sketch(opt.path_to_real_sketch)
        v_sketch = v_sketch.unsqueeze(0).unsqueeze(0)

        for s in range(14): 
            skip_indx = torch.LongTensor([s+1]) 
            view = v_real[:, skip_indx, :]
            x_mu = model.generate(x_real, v_sketch, view, skip_indx, 99)
            imglist.append(x_mu.squeeze(0).cpu().detach())
            i += 1

        plt.figure().suptitle('Generated', fontsize=20)
        show(make_grid(imglist))
        plt.xticks([])
        plt.yticks([])
        plt.show()

    if opt.mode == 'mental_rotation_plot_real_sketch':
        # A plot version of the mental rotation of a single, real sketch  
        # where the model is rendered from various view points
        # Need to provide information of what class is in the sketch
        # Here it is obtained from the test dataset from a model from the same class
        v_sketch = v_real[0][12]

        x_real = prepare_sketch(opt.path_to_real_sketch)
        
        v_real = v_sketch.unsqueeze(0).unsqueeze(0)

        phi = torch.arange(2*np.pi/11, 2*np.pi, 2*np.pi/11)
        theta = torch.arange(np.pi/11, np.pi, np.pi/11)

        v = get_viewpoint(phi, theta, v_real)
        block_rot = Camera_view(v)
        # Generate images
        imglist = []
        for theta_angle in theta:
            for phi_angle in phi:
                view = block_rot.plot_grid(theta_angle, phi_angle)
                x_mu = model.generate(x_real, v_real, view, 44, 99) 
                imglist.append(x_mu.squeeze(0).cpu().detach())
                
        plt.figure().suptitle('Predictions', fontsize=10)
        show(make_grid(imglist, nrow=len(phi)))
        plt.xlabel('Yaw')
        plt.ylabel('Pitch')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    if opt.mode == 'massive_plot':
        # Generate a batch of images and show them with
        # corresponding ground truths
        # Set batch size in dataloader
        x_mu = model.generate(x_real, v_real, query_view, skip_indx, 99)
        plt.figure().suptitle('Generated', fontsize=20)
        plt.xticks([])
        plt.yticks([]) 
        show(make_grid(x_mu.detach().cpu()))
        plt.show()

        plt.figure().suptitle('Ground Truth', fontsize=20)
        plt.xticks([])
        plt.yticks([]) 
        show(make_grid(critic_img_real.cpu()))
        plt.show()
 
    if opt.mode == 'bayesian_surprise':
        # Calculates and plots the Bayesian surprise
        # for a single model or multiple models depending on
        # if opt.multi_class is set to True or not
        bayesian_surprise(opt, test_loader, model)

    if opt.mode == 'perceptual_loss':
        # Calculates and plots the Perceptual loss
        # for a single model or multiple models depending on
        # if opt.multi_class is set to True or not
        perceptual_loss(opt, test_loader, model)

    if opt.mode == "sketches":
        # Plot a batch of sketches. Change batch size in dataloader
        x_mu = x_real[:, 0, :, :, :]
        plt.figure().suptitle('Sketches', fontsize=20)
        plt.xticks([])
        plt.yticks([])
        print(x_mu.size()) 
        show(make_grid(x_mu.detach().cpu()))
        plt.show()

parser = argparse.ArgumentParser(description='Results for Sketch to 3D model GQN')
parser.add_argument('--mode', type=str, default='single', help='Result to display among the following: single model')
parser.add_argument('--multi_class', type=str, help='Use multiple classes in Bayesian Surprise/Perceptual Loss or not (default: False)', default='False')
parser.add_argument('--batch_size', type=int, default='1', help='Batch size, larger than 1 only works for certain results')
parser.add_argument('--test_dataset', type=str, default='.', help='Path to test data')
parser.add_argument('--path_to_model', type=str, default='.', help='Path to trained model')
parser.add_argument('--path_to_real_sketch', type=str, default='.', help='Path to to real sketch')
opt = parser.parse_args()

path = opt.path_to_model

torch.cuda.set_device(0)

model = GQN(representation='pool', L=12, shared_core=True).cuda()

checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

model.cuda()

torch.manual_seed(2) 

path_test = opt.test_dataset 

test_data = ShapeNetDatasetSketch(path_test)
test_loader = data.DataLoader(test_data, batch_size = opt.batch_size, shuffle=True, drop_last = True)

main()
