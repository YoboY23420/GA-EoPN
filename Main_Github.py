import time
import os
import torch
from torch import optim
from datetime import datetime
import Losses_Github, Utils_Github
from Model_Github import Model
from data import dataloader

def main(dataset):
    if dataset == 'OASIS':
        train_dir = '/Medical_Image_Registration/3D_brain_MRI/affine_img/'
        valid_dir = '/Medical_Image_Registration/3D_brain_MRI/affine_seg/'
        loader_train = dataloader.torch_Dataloader_OASIS(train_dir, valid_dir, 'train', batch_size=1)
        loader_validation = dataloader.torch_Dataloader_OASIS(train_dir, valid_dir, 'test', batch_size=1)
        save_exp = '/saved_models/experiments_OASIS'
        dataset = 'OASIS'
    elif dataset == 'LPBA40':
        train_dir = '/Medical_Image_Registration/3D_LPBA40/affine_img/'
        valid_dir = '/Medical_Image_Registration/3D_LPBA40/affine_seg/'
        loader_train = dataloader.torch_Dataloader_LPBA40(train_dir, valid_dir, 'train', batch_size=1)
        loader_validation = dataloader.torch_Dataloader_LPBA40(train_dir, valid_dir, 'test', batch_size=1)
        save_exp = '/saved_models/experiments_LPBA40'
        dataset = 'LPBA40'
    elif dataset == 'IXI':
        train_dir = '/Medical_Image_Registration/3D_IXI_dataset/IXI_data/Train/'
        valid_dir = '/Medical_Image_Registration/3D_IXI_dataset/IXI_data/Test/'
        loader_train = dataloader.torch_Dataloader_IXI(train_dir, valid_dir, 'train', batch_size=1)
        loader_validation = dataloader.torch_Dataloader_IXI(train_dir, valid_dir, 'test', batch_size=1)
        save_exp = '/saved_models/experiments_IXI'
        dataset = 'IXI'

    timestamp = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
    if os.path.exists(save_exp):
        os.makedirs(os.path.join(save_exp, timestamp))
    else:
        os.makedirs(os.path.join(save_exp))
        os.makedirs(os.path.join(save_exp, timestamp))

    img_size = (160, 192, 224)
    loss_weights = [1, 1]
    lr = 0.0001
    epoch_start = 0
    max_epoch = 1

    model = Model(img_size)
    model.cuda()
    reg_model = Utils_Github.register_model(img_size, 'nearest')
    reg_model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    criterion = Losses_Github.NCC_vxm()
    criterions = [criterion]
    criterions += [Losses_Github.Grad3d(penalty='l2')]
    for epoch in range(epoch_start, max_epoch):
        step_t = 0
        for pair_t, mi_t, fi_t, ml_t, fl_t in loader_train:
            step_t += 1
            model.train()
            mi_t = mi_t.unsqueeze(0).cuda()
            fi_t = fi_t.unsqueeze(0).cuda()
            output = model(mi_t, fi_t)
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], fi_t) * loss_weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('\rEpoch {}<-->Image {} to {}<-->Iter {}/{}<-->Loss {:.4f}<-->Img Sim {:.4f}<-->Reg {:.4f}'.format(epoch, pair_t[0], pair_t[1], step_t, len(loader_train), loss.item(), loss_vals[0].item(), loss_vals[1].item()))

            if dataset == 'OASIS':
                total_iter = 64770
                testing_iter = 5000
            elif dataset == 'LPBA40':
                total_iter = 756
                testing_iter = 50
            elif dataset == 'IXI':
                total_iter = 162006
                testing_iter = 1
            if step_t == total_iter or step_t % testing_iter == 0:
                eval_dsc = Utils_Github.AverageMeter()
                eval_njd = Utils_Github.AverageMeter()
                eval_time = Utils_Github.AverageMeter()
                with torch.no_grad():
                    step_v = 0
                    for pair_v, mi_v, fi_v, ml_v, fl_v in loader_validation:
                        step_v += 1
                        model.eval()
                        mi_v = mi_v.unsqueeze(0).cuda()
                        fi_v = fi_v.unsqueeze(0).cuda()
                        ml_v = ml_v.unsqueeze(0).cuda()
                        fl_v = fl_v.unsqueeze(0).cuda()
                        time_start = time.time()
                        output = model(mi_v, fi_v)
                        time_end = time.time()
                        def_out = reg_model([ml_v, output[1]])
                        dsc = Utils_Github.dice_val_ROI(def_out.long(), fl_v.long(), dataset=dataset)
                        eval_dsc.update(dsc.item(), n=1)
                        njd = Utils_Github.jacobian_determinant_vxm(output[1].detach().cpu().numpy().squeeze())
                        eval_njd.update(njd.item(), n=1)
                        time_usage = time_end - time_start
                        eval_time.update(time_usage, n=1)
                        print('\rEpoch {}<-->Image {} to {}<-->Iter {}/{}<-->AvgDSC {:.4f}<-->AvgNJD {:.4f}<-->AvgTime {:.4f}'.format(epoch, pair_v[0], pair_v[1], step_v, len(loader_validation), dsc, njd, time_usage), end='')
                torch.save(model.state_dict(), '{}/{}_Dice{:.4f}_NJD{:.4f}_Time{:.4f}.pth'.format(os.path.join(save_exp, timestamp), step_t, eval_dsc.avg, eval_njd.avg, eval_time.avg))
                print('Step {}<-->Dice{:.4f}_NJD{:.4f}_Time{:.4f}'.format(step_t, eval_dsc.avg, eval_njd.avg, eval_time.avg))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    dataset = 'OASIS'
    main(dataset)