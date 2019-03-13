from load_data import get_data
from arrange_data import arrange_data
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage, AnnotationBbox)
import torch
import numpy as np

def bayesian_surprise(opt, test_loader, model):
    # Calculates the Bayesian Surprise for one or several models
    if opt.multi_class == 'True':
        maximum_viewpoints = 6
        all_means = [0,0,0,0,0,0]
        all_errors = [0,0,0,0,0,0]
        nr_models = 50
        m = 1             
        M = 15
        # Get random models
        for models in tqdm(range(500, 500+nr_models)):
            x = get_data(test_loader, models) 
            
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
                kl_track = 0
                samples = 1000
                KL_measurements = []
           
                for j in range(0, samples):
                    kl_test = model.kl_divergence(seen_images, seen_viewp, held_out_view, held_out_real_ver, skip_indx, i).detach()
                    KL_measurements.append(kl_test)
                    kl_track += kl_test

                mean = (kl_track/samples).cpu().numpy() 
                deviation_from_mean = mean*np.ones_like(KL_measurements)-KL_measurements
                squared_deviation = deviation_from_mean**2
                sum_squared_dev = np.sum(squared_deviation)
                div_sum_by_sample_size = sum_squared_dev/(samples-1)
                standard_deviation = np.sqrt(div_sum_by_sample_size) 
                standard_error = standard_deviation/np.sqrt(samples)

                all_means[i] += mean[0]
                all_errors[i] += standard_error
                
        means = [x/nr_models for x in all_means]
        errors = [x/nr_models for x in all_errors]
        print("Mean: ", means)
        print("Errors: ", errors)
        i_s = range(0, 6)
        fig, ax = plt.subplots()
        ax.errorbar(i_s, means, yerr=errors, fmt='o')
        ax.set_ylabel('Surprise (bits)', fontsize=15)
    else:
        
        x = get_data(test_loader,22)

        data_tmp_sketch = x[0]
        data_tmp_real = x[1]
        
        maximum_viewpoints = 6
        M = 15
        skip_indx = 11
 
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

        sz = 11
        fig, ax = plt.subplots(figsize=(sz, sz))
        imagebox = OffsetImage(critic_img_real.cpu().squeeze(), cmap='Greys_r', zoom=2)
        imagebox.image.axes = ax
        fig_x = 5
        fig_y = 2

        for i in tqdm(range(0, maximum_viewpoints)):
            
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
            kl_track = 0
            samples = 1000
            KL_measurements = []
       
            for j in range(0, samples):

                kl_test = model.kl_divergence(seen_images, seen_viewp, held_out_view, held_out_real_ver, skip_indx, i).detach()
                KL_measurements.append(kl_test)
                kl_track += kl_test

            mean = (kl_track/samples).cpu().numpy() 
            deviation_from_mean = mean*np.ones_like(KL_measurements)-KL_measurements
            squared_deviation = deviation_from_mean**2
            sum_squared_dev = np.sum(squared_deviation)
            div_sum_by_sample_size = sum_squared_dev/(samples-1)
            standard_deviation = np.sqrt(div_sum_by_sample_size)
            standard_error = standard_deviation/np.sqrt(samples)
            means.append(mean[0])
            errors.append(standard_error)
            i_s.append(i)

        ids = torch.Tensor(range(maximum_viewpoints)).long()
        img_to_plot = x_real[0,ids,:,:]

        ax.errorbar(i_s, means, yerr=errors, fmt='o')
        ax.set_ylabel('Surprise (bits)', fontsize=15)

        grtr = held_out_observation[0][0]

        newax = fig.add_axes([0.57, 0.55, 0.3, 0.3], anchor='NE')
        newax.imshow(grtr, cmap='Greys_r')
        newax.axis('off')
        newax.set_title('Held out observation')

        # Annotate the 2nd position with an image
        zm = 1.2
        imagebox = OffsetImage(img_to_plot[0][0], zoom=zm, cmap='Greys_r')
        imagebox.image.axes = ax

        ab = AnnotationBbox(imagebox, (1,0),
                            xybox=(0, -7),
                            xycoords=("data", "axes fraction"),
                            boxcoords="offset points",
                            box_alignment=(.5, 1),
                            bboxprops={"edgecolor" : "none"})

        ax.add_artist(ab)

        imagebox = OffsetImage(img_to_plot[1][0], zoom=zm, cmap='Greys_r')
        imagebox.image.axes = ax

        ab = AnnotationBbox(imagebox, (2,0),
                            xybox=(0, -7),
                            xycoords=("data", "axes fraction"),
                            boxcoords="offset points",
                            box_alignment=(.5, 1),
                            bboxprops={"edgecolor" : "none"})

        ax.add_artist(ab)

        imagebox = OffsetImage(img_to_plot[2][0], zoom=zm, cmap='Greys_r')
        imagebox.image.axes = ax

        ab = AnnotationBbox(imagebox, (3,0),
                            xybox=(0, -7),
                            xycoords=("data", "axes fraction"),
                            boxcoords="offset points",
                            box_alignment=(.5, 1),
                            bboxprops={"edgecolor" : "none"})

        ax.add_artist(ab)

        imagebox = OffsetImage(img_to_plot[3][0], zoom=zm, cmap='Greys_r')
        imagebox.image.axes = ax

        ab = AnnotationBbox(imagebox, (4,0),
                            xybox=(0, -7),
                            xycoords=("data", "axes fraction"),
                            boxcoords="offset points",
                            box_alignment=(.5, 1),
                            bboxprops={"edgecolor" : "none"})

        ax.add_artist(ab)

        imagebox = OffsetImage(img_to_plot[4][0], zoom=zm, cmap='Greys_r')
        imagebox.image.axes = ax

        ab = AnnotationBbox(imagebox, (5,0),
                            xybox=(0, -7),
                            xycoords=("data", "axes fraction"),
                            boxcoords="offset points",
                            box_alignment=(.5, 1),
                            bboxprops={"edgecolor" : "none"})

        ax.add_artist(ab)

    plt.show()
    
    #fig.savefig('path/to/save/img', format='png')