import argparse
import datetime
import os
import numpy as np
import datetime
import time
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from scheduler import AnnealingStepLR
from model import GQN
from torch.utils import data
from arrange_data import arrange_data
from shape_net_dataset_sketch import ShapeNetDatasetSketch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sketch to 3D model Implementation')
    parser.add_argument('--gradient_steps', type=int, default=2*10**6, help='number of gradient steps to run (default: 2 million)')
    parser.add_argument('--batch_size', type=int, default=36, help='size of batch (default: 36)')
    parser.add_argument('--train_data_dir', type=str, help='location of training data', \
                        default="/workspace/dataset/train")
    parser.add_argument('--test_data_dir', type=str, help='location of test data', \
                        default="/workspace/dataset/test")
    parser.add_argument('--log_dir', type=str, help='location of log', default='/workspace/logs/')
    parser.add_argument('--log_interval', type=int, help='interval number of steps for logging', default=100)
    parser.add_argument('--save_interval', type=int, help='interval number of steps for saveing models', default=10000)
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--device_ids', type=int, nargs='+', help='list of CUDA devices (default: [0])', default=[0])
    parser.add_argument('--representation', type=str, help='representation network (default: pool)', default='pool')
    parser.add_argument('--layers', type=int, help='number of generative layers (default: 12)', default=12)
    parser.add_argument('--shared_core', type=bool, \
                        help='whether to share the weights of the cores across generation steps (default: False)', \
                        default=False)
    parser.add_argument('--seed', type=int, help='random seed (default: None)', default=None)
    parser.add_argument('--continue_training', type=str, help = 'continue training on old model(default: False)', default='False')
    parser.add_argument('--path_to_model', type=str, help='path to saved model', default='.')
    args = parser.parse_args()

    device = f"cuda:{args.device_ids[0]}" if torch.cuda.is_available() else "cpu"
    
    # Seed
    if args.seed!=None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # Dataset directory
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir

    # Number of workers to load data
    num_workers = args.workers

    # Log
    log_interval_num = args.log_interval
    save_interval_num = args.save_interval
    log_dir = args.log_dir
    try:
        os.mkdir(log_dir)
    except FileExistsError:
        pass

    # TensorBoardX
    writer = SummaryWriter(log_dir)

    path_train =  args.train_data_dir
    path_test = args.test_data_dir

    train_data = ShapeNetDatasetSketch(path_train)
    train_loader = data.DataLoader(train_data, batch_size = args.batch_size, shuffle=True, drop_last = True)

    test_data = ShapeNetDatasetSketch(path_test)
    test_loader = data.DataLoader(test_data, batch_size = args.batch_size, shuffle=True, drop_last = True)

    # Pixel standard-deviation
    sigma_i, sigma_f = 2.0, 0.7
    sigma = sigma_i

    # Number of scenes over which each weight update is computed
    B = args.batch_size
    
    # Number of generative layers
    L =args.layers

    # Maximum number of training steps
    S_max = args.gradient_steps

    # Define model
    model = GQN(representation=args.representation, L=L, shared_core=args.shared_core).to(device)
    if args.continue_training == 'False':
        if len(args.device_ids) > 1:
            model = nn.DataParallel(model, device_ids=args.device_ids)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08)
        scheduler = AnnealingStepLR(optimizer, mu_i=5e-5, mu_f=5e-6, n=1.6e6)
        t_start = 0
    else:
        path = args.path_to_model

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = AnnealingStepLR(optimizer, mu_i=5e-4, mu_f=5e-5, n=1.6e6)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        sigma = checkpoint['sigma']
        sigma_i = sigma
        t_start = checkpoint['time_step']
        model.train()

    model.to(device)

    train_iter = iter(train_loader)
    test_iter = iter(test_loader)

    # Training Iterations
    for t in tqdm(range(t_start, S_max)):
        try:
            data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data = next(train_iter)

        data_tmp_sketch = data[0]
        data_tmp_real = data[1]

        M = np.random.randint(1, 16)
        x_sketch, v_sketch, query_view, skip_indx, critic_img_sketch = arrange_data(data_tmp_sketch, M, True, 0)
        x_real, v_real, query_view_real, skip_indx, critic_img_real = arrange_data(data_tmp_real, M, False, skip_indx)

        x_sketch = x_sketch.to(device)
        v_sketch = v_sketch.to(device)
        query_view_sketch = query_view.to(device)
        critic_img_real = critic_img_real.to(device)

        elbo = model(x_sketch, v_sketch, query_view_sketch, critic_img_real, sigma, skip_indx)
        
        # Logs
        writer.add_scalar('loss/train_loss_elbo', -elbo.mean(), t)
             
        with torch.no_grad():

            try:
                test_iter_data = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                test_iter_data = next(test_iter)

            # Write logs to TensorBoard
            if t % log_interval_num == 0:
                data_tmp_sketch = test_iter_data[0]
                data_tmp_real = test_iter_data[1]

                M = 15
                x_sketch, v_sketch, query_view, skip_indx, critic_Img_sketch = arrange_data(data_tmp_sketch, M, True, 0)
                x_real, v_real, query_view_real, skip_indx, critic_img_real = arrange_data(data_tmp_real, M, False, skip_indx)

                x_data_test = x_sketch.to(device)
                v_data_test = v_sketch.to(device)
                query_view_sketch = query_view.to(device)
                critic_img_real = critic_img_real.to(device)

                elbo_test = model(x_data_test, v_data_test, query_view_sketch, critic_img_real, sigma, skip_indx)
                
                if len(args.device_ids)>1:
                    kl_test = model.module.kl_divergence(x_data_test, v_data_test, query_view_sketch, critic_img_real, skip_indx, 99) 
                    x_q_rec_test = model.module.reconstruct(x_data_test, v_data_test, query_view_sketch, critic_img_real, skip_indx)
                    x_q_hat_test = model.module.generate(x_data_test, v_data_test, query_view_sketch, skip_indx, 99)
                else:
                    kl_test = model.kl_divergence(x_data_test, v_data_test, query_view_sketch, critic_img_real, skip_indx, 99) 
                    x_q_rec_test = model.reconstruct(x_data_test, v_data_test, query_view_sketch, critic_img_real, skip_indx)
                    x_q_hat_test = model.generate(x_data_test, v_data_test, query_view_sketch, skip_indx, 99)

                writer.add_scalar('loss/test_loss', -elbo_test.mean(), t)
                writer.add_scalar('loss/test_kl', kl_test.mean(), t)
                writer.add_image('loss/test_ground_truth', make_grid(critic_img_real, 6, pad_value=1), t)
                writer.add_image('loss/test_reconstruction', make_grid(x_q_rec_test, 6, pad_value=1), t)
                writer.add_image('loss/test_generation', make_grid(x_q_hat_test, 6, pad_value=1), t)

            if t % save_interval_num == 0:
                # Save model
                ts = time.time()
                st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
                torch.save({'model_state_dict' : model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'time_step': t,
                            'sigma':  sigma}, 
                            log_dir + '/' + st + "model-{}.pt".format(t))


        # Compute empirical ELBO gradients
        (-elbo.mean()).backward()

        if t % 100 == 0:
            gradr = [torch.max(p.grad) if p.grad is not None else 0 for p in list(model.parameters())]
            gradr = max(gradr)

            larg_lr_crit = [x["lr"] for x in list(optimizer.param_groups)]
            larg_lr_crit = max(larg_lr_crit)

            writer.add_scalar('loss/grad', gradr, t)
            writer.add_scalar("loss/lr", larg_lr_crit, t)

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

        # Update optimizer state
        scheduler.step()

        # Pixel-variance annealing
        sigma = max(sigma_f + (sigma_i - sigma_f)*(1 - t/(2e5)), sigma_f)
        writer.add_scalar('loss/sigma', sigma, t)        
        
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
    torch.save({'model_state_dict' : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'time_step': t,
                'sigma':  sigma},
                log_dir + '/' + st + "model-{}.pt".format(t))

    writer.close()
