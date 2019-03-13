from load_data import get_data
from arrange_data import arrange_data
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.models import vgg19
from torch.autograd import Variable, grad

class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        # Extracts features at the 11th layer
        self.feature_extractor = torch.nn.Sequential(*list(vgg19_model.features.children())[:12])

    def forward(self, img):
        out = self.feature_extractor(img)
        return out

def perceptual_loss(opt, test_loader, model):

    feature_extractor = FeatureExtractor()
    criterion_content = torch.nn.L1Loss()
    feature_extractor = feature_extractor.cuda()

    if opt.multi_class == 'True':
        maximum_viewpoints = 6
        all_means = [0,0,0,0,0,0]
        all_errors = [0,0,0,0,0,0]
        nr_models = 50 
        random_start_id = 1500
        m = 1             
        M = 15
        for models in tqdm(range(random_start_id, random_start_id+nr_models)):
            x = get_data(test_loader,models) 
            
            data_tmp_sketch = x[0]
            data_tmp_real = x[1]

            # Just any view point not within range of maximum_viewpoints
            skip_indx = 9
     
            x_real, v_real, query_view, skip_indx, critic_img = arrange_data(data_tmp_sketch, M, False, skip_indx) # Set to False to control view point
            x_real_real, v_real_real, query_view_real, skip_indx, critic_img_real = arrange_data(data_tmp_real, M, False, skip_indx)

            x_real = x_real.float()
            x_real = x_real.cuda()
            v_real = v_real.cuda()
            query_view = query_view.cuda()
            x_real_real = x_real_real.cuda()

            critic_img_real = critic_img_real.cuda()
            errors = []
            means = []
            i_s = []

            for i in range(0, maximum_viewpoints):
                
                if i != 0:
                    ids = torch.Tensor(range(i)).long()
                    seen_images = x_real[:,ids,:,:].cuda()
                    seen_viewp = v_real[:,ids,:].cuda()
                
                else:
                    seen_images = torch.zeros_like(x_real[:,1,:,:]).cuda()
                    seen_viewp = torch.zeros_like(v_real[:,1,:]).cuda()

                held_out_observation = x_real[:,skip_indx,:,:].cuda()
                held_out_view = v_real[:,skip_indx,:].cuda()

                held_out_real_ver = x_real_real[:,skip_indx,:,:].cuda()
                percept_track = 0
                samples = 1000
                perceptual_measurements = []
           
                for j in range(0, samples):
                    x_mu = model.generate(seen_images, seen_viewp, held_out_view, skip_indx, i) # i is usually not part of generate, also, set prior to sample again 
                    gen_features = feature_extractor(x_mu.repeat(1,3,1,1)) # Fully convolutional so can take any dim I think
                    real_features = Variable(feature_extractor(held_out_real_ver.repeat(1,3,1,1)).data, requires_grad=False) 
                    loss_content =  criterion_content(gen_features, real_features).cpu().detach().numpy()

                    perceptual_measurements.append(loss_content)
                    percept_track += loss_content

                mean = (percept_track/samples)
                deviation_from_mean = mean*np.ones_like(perceptual_measurements)-perceptual_measurements
                squared_deviation = deviation_from_mean**2
                sum_squared_dev = np.sum(squared_deviation)
                div_sum_by_sample_size = sum_squared_dev/(samples-1)
                standard_deviation = np.sqrt(div_sum_by_sample_size)
                standard_error = standard_deviation/np.sqrt(samples)
                all_means[i] += mean
                all_errors[i] += standard_error
               
        means = [x/nr_models for x in all_means]
        errors = [x/nr_models for x in all_errors]
        print("Mean: ", means)
        print("Errors: ", errors)
        i_s = range(0, maximum_viewpoints)
        fig, ax = plt.subplots()

    else:

        x = get_data(test_loader,22)

        data_tmp_sketch = x[0]
        data_tmp_real = x[1]

        maximum_viewpoints = 6
        M = 15
        skip_indx = 9
 
        x_real, v_real, query_view, skip_indx, critic_img = arrange_data(data_tmp_sketch, M, False, skip_indx) # Set to False to control view point
        x_real_real, v_real_real, query_view_real, skip_indx, critic_img_real = arrange_data(data_tmp_real, M, False, skip_indx)

        x_real = x_real.float()
        x_real = x_real.cuda()
        v_real = v_real.cuda()
        query_view = query_view.cuda()
        x_real_real = x_real_real.cuda()

        critic_img_real = critic_img_real.cuda()
        errors = []
        means = []
        i_s = []

        save_img = []

        fig, ax = plt.subplots()

        for i in range(0, maximum_viewpoints):
            
            if i != 0:
                ids = torch.Tensor(range(i)).long()

                seen_images = x_real[:,ids,:,:].cuda()
                seen_viewp = v_real[:,ids,:].cuda()
                save_seen_images = x_real[:,4,:,:]
            
            else:
                seen_images = torch.zeros_like(x_real[:,1,:,:]).cuda()
                seen_viewp = torch.zeros_like(v_real[:,1,:]).cuda()

            held_out_observation = x_real[:,skip_indx,:,:].cuda()
            held_out_view = v_real[:,skip_indx,:].cuda()

            held_out_real_ver = x_real_real[:,skip_indx,:,:].cuda()
            percept_track = 0
            samples = 1000
            perceptual_measurements = []
       
            for j in range(0, samples):
                x_mu = model.generate(seen_images, seen_viewp, held_out_view, skip_indx, i)
                
                gen_features = feature_extractor(x_mu.repeat(1,3,1,1))
                real_features = Variable(feature_extractor(held_out_real_ver.repeat(1,3,1,1)).data, requires_grad=False) 
                loss_content =  criterion_content(gen_features, real_features).cpu().detach().numpy()
                perceptual_measurements.append(loss_content)
                percept_track += loss_content

            mean = (percept_track/samples) 
            deviation_from_mean = mean*np.ones_like(perceptual_measurements)-perceptual_measurements
            squared_deviation = deviation_from_mean**2
            sum_squared_dev = np.sum(squared_deviation)
            div_sum_by_sample_size = sum_squared_dev/(samples-1)
            standard_deviation = np.sqrt(div_sum_by_sample_size)
            standard_error = standard_deviation/np.sqrt(samples)
            print("Standard deviation: ", standard_deviation)
            print("Standard error: ", standard_error)
            print("Mean: ", mean)
            means.append(mean)
            errors.append(standard_error)
            i_s.append(i)

    ax.errorbar(i_s, means, yerr=errors, fmt='o')
    ax.set_ylabel('Perceptual Loss', fontsize=15)
    ax.xaxis.set_ticks(np.arange(0, 6, 1))
    plt.show()