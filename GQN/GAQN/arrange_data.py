import torch

def arrange_data(data_tmp, M):
    # Extracts batch of data from dataset loader
    x_tmp = [x[0][0] for x in data_tmp]
    v_tmp = [v[1][0] for v in data_tmp]
    rand_idx = torch.LongTensor(1).random_(0, 15)

    x_tmp = torch.stack(x_tmp)
    v_tmp = torch.stack(v_tmp)

    v_tmp_xyz = v_tmp.narrow(dim=2, start=0, length=3)
    v_tmp_jawpitch = v_tmp.narrow(dim=2, start=3, length=2)
    v_tmp_jawpitch_cosed = torch.cos(v_tmp_jawpitch)
    v_tmp_jawpitch_sined = torch.sin(v_tmp_jawpitch)
    v_tmp = torch.cat([torch.cat([v_tmp_xyz, v_tmp_jawpitch_cosed], dim=2), v_tmp_jawpitch_sined], dim=2)

    q_tmp = v_tmp[:,rand_idx, :]

    v_real_query = q_tmp
    critic_img = x_tmp[:,rand_idx, :, :, :]
    x_real = x_tmp.narrow(dim= 1, start=0, length=M)
    v_real = v_tmp.narrow(dim= 1, start=0, length=M)
    x_real = x_real.permute(0, 1, 4, 2, 3)
    critic_img = critic_img.permute(0, 1, 4, 2, 3).squeeze()

    return x_real, v_real, v_real_query, rand_idx, critic_img

def arrange_data_generation(data_tmp, M):
    # Extracts scene from dataset loader
    x_tmp = data_tmp[0][0]
    v_tmp = data_tmp[1][0]
    rand_idx = torch.LongTensor(1).random_(0, 15)

    v_tmp_xyz = v_tmp.narrow(dim=1, start=0, length=3)
    v_tmp_jawpitch = v_tmp.narrow(dim=1, start=3, length=2)
    v_tmp_jawpitch_cosed = torch.cos(v_tmp_jawpitch)
    v_tmp_jawpitch_sined = torch.sin(v_tmp_jawpitch)
    v_tmp = torch.cat([torch.cat([v_tmp_xyz, v_tmp_jawpitch_cosed], dim=1), v_tmp_jawpitch_sined], dim=1)

    indx = torch.tensor([rand_idx])

    q_tmp = torch.index_select(v_tmp, dim=0, index=indx)

    v_real_query = q_tmp
    critic_img = x_tmp[rand_idx, :, :, :]
    x_real = x_tmp.narrow(dim= 0, start=0, length=M)
    v_real = v_tmp.narrow(0, 0, M)
    x_real = x_real.permute(0, 3, 1, 2)
    critic_img = critic_img.permute(0, 3, 1, 2)

    return x_real, v_real, v_real_query, rand_idx, critic_img