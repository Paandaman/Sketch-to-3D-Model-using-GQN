from gqn_dataset import *
from representator import Representator
from critic import Critic
from generator import Generator
from torch.autograd import Variable, grad
from arrange_data import arrange_data
from tensorboardX import SummaryWriter
from torch.utils import data
import argparse 
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch
import datetime
import time
import os
import numpy as np

parser = argparse.ArgumentParser(description='Generative Adversarial Query Networks')
parser.add_argument('--gradient_steps', type=int, default=2*10**6, help='number of gradient steps to run (default: 2 million)')
parser.add_argument('--batch_size', type=int, default=8, help='size of batch (default: 8)')
parser.add_argument('--epochs', type=int, default=40, help='epochs (default: 40)')
parser.add_argument('--upd_discr', type=int, default=1, help='number of discriminator updates per generator update (default: 1)')
parser.add_argument('--train_data_dir', type=str, help='location of training data', default="/workspace/dataset/train")
parser.add_argument('--test_data_dir', type=str, help='location of test data', default="/workspace/dataset/test")
parser.add_argument('--log_dir', type=str, help='location of log dir', default='/workspace/logs/')
opt = parser.parse_args()

torch.cuda.set_device(0)

m = opt.batch_size
latent_size = 100
epochs = opt.epochs
upd_discr = opt.upd_discr

tf.enable_eager_execution()
path = opt.train_data_dir
train_data = GqnDataset(path)
trainloader = data.DataLoader(train_data, shuffle=True)

latent_distr = torch.distributions.normal.Normal(0, 1)
uniform_distr = torch.distributions.uniform.Uniform(0, 1)

# Networks      
crit = Critic()
gen = Generator()
rep = Representator()
rep_gen = Representator()
crit.cuda()
gen.cuda()
rep.cuda()
rep_gen.cuda()

optimizer = torch.optim.Adam(crit.parameters(), lr = 0.0004, betas=(0.0, 0.9)) # TTUR
optimizer_gen = torch.optim.Adam(gen.parameters(), lr = 0.001, betas=(0.0, 0.9))
optimizer_rep = torch.optim.Adam(rep.parameters(), lr = 0.0001, betas=(0.0, 0.9))
optimizer_rep_gen = torch.optim.Adam(rep_gen.parameters(), lr = 0.001, betas=(0.0, 0.9))

# exponentially decaying learning rate
scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_gen, gamma=0.999)
scheduler_r = torch.optim.lr_scheduler.ExponentialLR(optimizer_rep, gamma=0.999)
scheduler_r_gen = torch.optim.lr_scheduler.ExponentialLR(optimizer_rep_gen, gamma=0.999)

discr_loss = []
gen_loss = []
tmp_discr_loss = 0
fin_time = 0
noise_gamma = 0.1

output_dir = opt.log_dir
os.mkdir(output_dir)
writer = SummaryWriter(output_dir)

def save_model(crit, gen, rep, rep_gen, optimizer, optimizer_gen, optimizer_rep, optimizer_rep_gen, fin_time):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')

    torch.save({
                'Critic_state_dict': crit.state_dict(),
                'Gen_state_dict': gen.state_dict(),
                'Rep_state_dict': rep.state_dict(),
                'Rep_gen_state_dict': rep_gen.state_dict(),
                'optimizerCritic_state_dict': optimizer.state_dict(),
                'optimizerGen_state_dict': optimizer_gen.state_dict(),
                'optimizerRep_state_dict': optimizer_rep.state_dict(),
                'optimizerRep_gen_state_dict': optimizer_rep_gen.state_dict(),
                'fin_time': (fin_time) 
                }, output_dir + st)

def convert_img_for_display(gen_x):
    OldMin = -1
    OldMax = 1
    NewMin = 0
    NewMax = 1
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    img = ((gen_x - OldMin)*NewRange/OldRange+NewMin)
    return img

def calc_gradient_penalty(netCrit, real_data, fake_data, query_view, representation): 
    """Computes the gradient penalty for the Improved Wasserstein loss using data from the dataset and the
       generator. Currently not used.
    """
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous().view(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data) 
    interpolates = interpolates.cuda()                          
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netCrit(interpolates, query_view, representation)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,grad_outputs=torch.ones(disc_interpolates.size()).cuda(),create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

for epoch in range(epochs):
    for t, data in enumerate(trainloader, 0):
        # The data is loaded into chunks of 2000 scenes
        batch_nr = 0
        while (2000-batch_nr) > 0:
            tmp_discr_loss = 0
            tmp_discr_grad = 0

            # Train Critic(discriminator)
            for i in range(upd_discr):
                if (2000- batch_nr) > m:
                    data_tmp = data[batch_nr:batch_nr+m]
                else:
                    break

                optimizer.zero_grad()
                optimizer_rep.zero_grad()

                M = np.random.randint(1, 16)
                x_real, v_real, query_view, skip_indx, critic_img = arrange_data(data_tmp, M)
                skip_indx = skip_indx.data[0]

                x_real = x_real.float()
                x_real = x_real.cuda()
                critic_img = critic_img.float()
                critic_img = critic_img.cuda()
                v_real = v_real.cuda()
                query_view = query_view.cuda()
                 
                zeros = torch.zeros(m,1).cuda() 
                latent_sample = latent_distr.sample(torch.Size([latent_size, m])).view(m, latent_size)
                loss = 0

                # generate samples
                x_lat = latent_sample.view(-1, latent_size).cuda()
                
                r = torch.zeros(m, 256, 1, 1).cuda()                
                r_gen = torch.zeros(m, 256, 1, 1).cuda()

                for view in range(0, M):
                    if view != skip_indx:
                        x_s = x_real[:,view, :, :, :]
                        v_s = v_real[:,view, :]
                        r += rep(x_s, v_s)
                        r_gen += rep_gen(x_s, v_s)

                # Normalize
                r = torch.div(r, M)
                r_gen = torch.div(r_gen, M)

                # Condition on neural representation and query view
                gen_data = gen(x_lat, r_gen, query_view)

                # Instance noise
                noise1 = torch.randn(critic_img.size()).cuda()
                critic_img = critic_img + noise_gamma*noise1

                eval_real = crit(critic_img, query_view, r)
                x_gen = gen_data.detach()

                noise2 = torch.randn(x_gen.size()).cuda()
                x_gen = x_gen + noise_gamma*noise2

                # Let Critic discriminate between images together with
                # query_view and representation
                eval_gen = crit(x_gen, query_view, r)

                ##grad_pen = calc_gradient_penalty(crit, critic_img, x_gen, query_view, r)

                loss = -torch.min(zeros, -1 + eval_real) - torch.min(zeros, -1 - eval_gen) #+ grad_pen*lamb
                loss = loss.mean()
                
                loss.backward()
 
                optimizer.step()
                optimizer_rep.step()

                tmp_discr_loss += loss.detach()

                batch_nr += m

            writer.add_scalar("loss/crit", tmp_discr_loss/upd_discr, fin_time)

            if t % 1 == 0:
                grads = [torch.max(p.grad) if p.grad is not None else 0 for p in list(crit.parameters())]
                grads = max(grads).detach()
                gradr = [torch.max(p.grad) if p.grad is not None else 0 for p in list(rep.parameters())]
                gradr = max(gradr).detach()
                writer.add_scalar('loss/Crit grad', grads, fin_time)
                writer.add_scalar('loss/Crit_Repr grad', gradr, fin_time)

            # Train Generator
            if (2000- batch_nr) > m:
                data_tmp = data[batch_nr:batch_nr+m]
            else:
                break

            optimizer_gen.zero_grad()
            optimizer_rep_gen.zero_grad()

            M = np.random.randint(1, 16)
            x_real, v_real, query_view, skip_indx, critic_img2 = arrange_data(data_tmp, M)
            skip_indx = skip_indx.data[0]

            x_real = x_real.float()
            x_real = x_real.cuda()
            v_real = v_real.cuda()
            query_view = query_view.cuda()
            skip_indx = skip_indx
  
            latent_sample = latent_distr.sample(torch.Size([latent_size, m])).view(m, latent_size)

            # generate samples
            x_lat = latent_sample.view(-1, latent_size).cuda()

            r = torch.zeros(m, 256, 1, 1).cuda()
             
            r_gen = torch.zeros(m, 256, 1, 1).cuda()

            for view in range(0, M):
                if view != skip_indx:
                    x_s = x_real[:,view, :, :, :]
                    v_s = v_real[:,view, :] 
                    r += rep(x_s, v_s)
                    r_gen += rep_gen(x_s, v_s)

            # Normalize
            r = torch.div(r, M)
            r_gen = torch.div(r_gen, M)

            # generate samples
            gen_data2 = gen(x_lat, r_gen, query_view)
            eval_gen2 = crit(gen_data2, query_view, r)

            loss2 = -eval_gen2.mean()
            loss2.backward()
            if t % 1 == 0:
                gradc = [torch.max(p.grad) if p.grad is not None else 0 for p in list(gen.parameters())]
                gradc = max(gradc).detach()
                gradr = [torch.max(p.grad) if p.grad is not None else 0 for p in list(rep_gen.parameters())]
                gradr = max(gradr).detach()
                writer.add_scalar('loss/Gen grad', gradc, fin_time)
                writer.add_scalar('loss/Repr grad', gradr, fin_time)

            optimizer_gen.step()
            optimizer_rep_gen.step()
            writer.add_scalar("loss/gen",loss2.detach(), fin_time)
            fin_time += 1
            batch_nr += m
        
        if t % 2 == 0:
            print(t)
            print(epoch)
            gen_x = gen_data2[0,:,:,:].cpu()
            gen_x = convert_img_for_display(gen_x) 
            name = "Image" + str(fin_time) + "_viewp_" + str(M)
            real_xx = convert_img_for_display(critic_img2.cpu()[0])

            r = r[0].mean(1).unsqueeze(0)
            r = F.sigmoid(r)
        
            writer.add_image(name, gen_x, fin_time)
            writer.add_image(name+"real", real_xx, fin_time)

        if t % 100 == 0 and epoch > 0:
            crit.gamma = min(crit.gamma+0.05, 1) # Gradually increase Self Attention
            gen.gamma = min(gen.gamma+0.05, 1)
            noise_gamma = max(noise_gamma-0.005, 0)
            scheduler_d.step()
            scheduler_g.step()
            scheduler_r.step()
            scheduler_r_gen.step()
            larg_lr_crit = [x["lr"] for x in list(optimizer.param_groups)]
            larg_lr_crit = max(larg_lr_crit)
            larg_lr_gen = [x["lr"] for x in list(optimizer_gen.param_groups)]
            larg_lr_gen = max(larg_lr_gen)
            writer.add_scalar("loss/crit_lr", larg_lr_crit, fin_time)
            writer.add_scalar("loss/crit_lr_gen", larg_lr_gen, fin_time)
            writer.add_scalar("loss/crit_gamma",crit.gamma, fin_time)
            writer.add_scalar("loss/gen_gamma",gen.gamma, fin_time)

        if t % 100 == 0:
            save_model(crit, gen, rep, rep_gen, optimizer, optimizer_gen, optimizer_rep, optimizer_rep_gen, fin_time)


    if epoch % 2 == 0:
        save_model(crit, gen, rep, rep_gen, optimizer, optimizer_gen, optimizer_rep, optimizer_rep_gen, fin_time)


writer.export_scalars_to_json("./all_scalars.json")
writer.close()
save_model(crit, gen, rep,  rep_gen, optimizer, optimizer_gen, optimizer_rep, optimizer_rep_gen, fin_time)

