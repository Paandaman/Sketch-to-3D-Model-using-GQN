import torch

def arrange_data(data_tmp, M, gen_rand_indx, rand_idx):
    # Extracts a batch of data from the dataset loader
    x_tmp = []
    v_tmp = []
    for data in data_tmp:
        x_tmp.append(torch.stack([x[0] for x in data]))
        v_tmp.append(torch.stack([v[1] for v in data]))
    
    if gen_rand_indx:
        rand_idx = torch.LongTensor(1).random_(0, 15)
    
    x_tmp = torch.stack(x_tmp) 
    v_tmp = torch.stack(v_tmp) 
  
    x_tmp = x_tmp.squeeze(0)
    v_tmp = v_tmp.squeeze(0)
    x_tmp = x_tmp.squeeze(2)
    v_tmp = v_tmp.squeeze(2)
    x_tmp = x_tmp.permute(1, 0, 2, 3, 4)
    v_tmp = v_tmp.permute(1, 0, 2)
    
    # Remove unecesary dimensions from black & white img
    x_tmp = x_tmp.narrow(dim=2, start=0, length=1)

    v_tmp_xyz = v_tmp.narrow(dim=2, start=0, length=3)
    v_tmp_jawpitch = v_tmp.narrow(dim=2, start=3, length=2)
    # Class condition in form of a class unique number
    v_tmp_classID = v_tmp.narrow(dim=2, start=5, length=1)
    # Get ID down to the same scale as the other features. 
    v_tmp_classID = torch.div(v_tmp_classID, 1000000) 
    v_tmp_jawpitch_cosed = torch.cos(v_tmp_jawpitch)
    v_tmp_jawpitch_sined = torch.sin(v_tmp_jawpitch)

    v_pitch_cosed = v_tmp_jawpitch_cosed.narrow(dim=2, start=0, length=1)
    v_pitch_sined = v_tmp_jawpitch_sined.narrow(dim=2, start=0, length=1)
    v_jaw_cosed = v_tmp_jawpitch_cosed.narrow(dim=2, start=0, length=1)
    v_jaw_sined = v_tmp_jawpitch_sined.narrow(dim=2, start=0, length=1)

    v_tmp = torch.cat([torch.cat([v_tmp_xyz, v_jaw_cosed], dim=2), v_jaw_sined], dim=2)
    v_tmp = torch.cat([torch.cat([v_tmp, v_pitch_cosed], dim=2), v_pitch_sined], dim=2)
    v_tmp = torch.cat([v_tmp, v_tmp_classID], dim=2)    

    q_tmp = v_tmp[:,rand_idx, :]

    v_real_query = q_tmp
    ground_truth = x_tmp[:,rand_idx, :, :, :]
    # Keep 0:M frames from scene
    x_real = x_tmp.narrow(dim= 1, start=0, length=M) 
    v_real = v_tmp.narrow(dim= 1, start=0, length=M)
    
    ground_truth = ground_truth.squeeze(dim=1)

    return x_real, v_real, v_real_query, rand_idx, ground_truth