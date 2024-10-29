import torch
import numpy as np
from datasets import *
from torch.utils.data import DataLoader
import evalMetrics as metrics
from torch import optim
from tensorboardX import SummaryWriter
import time
import datetime
import os
import sys
from atlas_models import SVF_resid
import SimpleITK as sitk
from atlas_utils import *
sys.path.append(os.path.realpath(".."))
import warnings
import argparse
import random
import torch.backends.cudnn as cudnn
from unigradicon import get_multigradicon

from icon_registration.mermaidlite import compute_warped_image_multiNC


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 2), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--reg-factor', default=20000.0, type=float, help='regularization factor')
parser.add_argument('--sim-factor', default=10.0, type=float, help='similarity factor')
parser.add_argument('--atlas-pair-sim-factor', default=0.0, type=float, help='pairwise similarity factor in atlas space')
parser.add_argument('--image-pair-sim-factor', default=0.0, type=float, help='pairwise similarity factor in image space')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--sim-loss', default='SSD', type=str, help='Similarity Loss to use.')
parser.add_argument('--save-per-epoch', default=10, type=int, help='number of epochs to save model.')


if __name__ == "__main__":

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    train_single_list, valid_single_list = get_train_valid_list_forward_atlas()

    max_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    loss_name = args.sim_loss
    best_score = 0.0

    reg_factor = args.reg_factor
    sim_factor = args.sim_factor
    atlas_pair_sim_factor = args.atlas_pair_sim_factor
    image_pair_sim_factor = args.image_pair_sim_factor

    using_affine_init = False

    SVFNet_train = OAI_Atlas_Opt_3D(train_single_list)
    SVFNet_train_dataloader = DataLoader(SVFNet_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    SVFNet_atlas_update = OAI_Atlas_Opt_3D(train_single_list)
    SVFNet_atlas_update_dataloader = DataLoader(SVFNet_atlas_update, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    SVFNet_val = OAI_Atlas_Opt_3D(valid_single_list)
    SVFNet_val_dataloader = DataLoader(SVFNet_val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    experiment_name = 'DHCP_UnigradIcon_' + str(loss_name) \
                      + '_affine_init_' + str(int(using_affine_init)) \
                      + '_seed_' + str(args.seed) \
                      + '_reg_' + str(reg_factor) \
                      + '_atlas_sim_' + str(sim_factor) \
                      + '_atlas_pair_sim_' + str(atlas_pair_sim_factor) \
                      + '_image_pair_sim_' + str(image_pair_sim_factor) \
                      + '_epoch_' + str(max_epochs) \
                      + '_network_lr_' + str(lr)
    train_model = get_multigradicon()
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        train_model.cuda(args.gpu)
    else:
        train_model.cuda()

    optimizer = optim.Adam(train_model.parameters(), lr=lr)

    train_model.train()
    now = datetime.datetime.now()
    now_date = "{:02d}{:02d}{:02d}".format(now.month, now.day, now.year)
    now_time = "{:02d}{:02d}{:02d}".format(now.hour, now.minute, now.second)
    writer = SummaryWriter(os.path.join('./logs', now_date, experiment_name + '_' + now_time))

    transform = tio.Compose([
        tio.Resize(175),
    ])

    init_sub = tio.Subject(
        img=tio.ScalarImage('atlas_image_initial.nii.gz'),
        label=tio.ScalarImage('atlas_labelmap_initial.nii.gz')
    )

    init_sub = transform(init_sub)
    atlas_tensor = init_sub['img'][tio.DATA]
    atlas_seg_tensor = init_sub['label'][tio.DATA]
    bilinear = Bilinear(zero_boundary=False)

    img_sz = np.array([175, 175, 175])
    batch_sz = batch_size
    train_identity_map = gen_identity_map(img_sz).unsqueeze(0).repeat(batch_sz, 1, 1, 1, 1).cuda(args.gpu)
    val_identity_map = gen_identity_map(img_sz).unsqueeze(0).cuda(args.gpu)

    for epoch in range(max_epochs):
        atlas_imgs = atlas_tensor.repeat(batch_sz, 1, 1, 1, 1).cuda(args.gpu)
        atlas_segs = atlas_seg_tensor.repeat(batch_sz, 1, 1, 1, 1).cuda(args.gpu)
        for i, (src_imgs, src_segs, src_ids) in enumerate(SVFNet_train_dataloader):
            global_step = epoch * len(SVFNet_train_dataloader) + (i + 1) * batch_size
            src_imgs, src_segs = src_imgs.cuda(args.gpu), src_segs.cuda(args.gpu)
            optimizer.zero_grad()

            _ = train_model(atlas_imgs, src_imgs)
            pos_deform_field = train_model.phi_AB_vectorfield
            neg_deform_field = train_model.phi_BA_vectorfield
            phi_AB = train_model.phi_AB_vectorfield - train_model.identity_map
            phi_BA = train_model.phi_BA_vectorfield - train_model.identity_map

            svf_warped_atlas_imgs = compute_warped_image_multiNC(
                atlas_imgs,
                train_model.phi_AB_vectorfield,
                train_model.spacing,
                0,
                zero_boundary=True
            )

            svf_warped_src_imgs = compute_warped_image_multiNC(
                src_imgs,
                train_model.phi_BA_vectorfield,
                train_model.spacing,
                0,
                zero_boundary=True
            )

            ## to evaluate in image space
            if image_pair_sim_factor != 0.0:
                sec_pos_deform_field = torch.flip(pos_deform_field, dims=[0])
                sec_src_imgs = torch.flip(src_imgs, dims=[0])
                svf_warped_src_imgs_in_image_space = bilinear(src_imgs, (bilinear(phi_BA, sec_pos_deform_field) + sec_pos_deform_field))


            ## loss
            sim_loss = get_sim_loss(svf_warped_atlas_imgs, src_imgs, loss_name)
            reg_loss = get_reg_loss(phi_BA)

            if atlas_pair_sim_factor != 0.0:
                atlas_pair_sim_loss = get_pair_sim_loss(svf_warped_src_imgs, loss_name)

            if image_pair_sim_factor != 0.0:
                image_pair_sim_loss = get_pair_sim_loss_image_space(svf_warped_src_imgs_in_image_space, sec_src_imgs, loss_name)

            if atlas_pair_sim_factor == 0.0 and image_pair_sim_factor == 0.0:
                loss = sim_factor * sim_loss + reg_factor * reg_loss
            elif atlas_pair_sim_factor != 0.0 and image_pair_sim_factor == 0.0:
                loss = sim_factor * sim_loss + reg_factor * reg_loss + atlas_pair_sim_factor * atlas_pair_sim_loss
            elif atlas_pair_sim_factor == 0.0 and image_pair_sim_factor != 0.0:
                loss = sim_factor * sim_loss + reg_factor * reg_loss + image_pair_sim_factor * image_pair_sim_loss
            elif atlas_pair_sim_factor != 0.0 and image_pair_sim_factor != 0.0:
                loss = sim_factor * sim_loss + reg_factor * reg_loss + atlas_pair_sim_factor * atlas_pair_sim_loss + image_pair_sim_factor * image_pair_sim_loss

            loss.backward()
            optimizer.step()
            writer.add_scalar('loss/training', loss.item(), global_step=global_step)
            if atlas_pair_sim_factor == 0.0 and image_pair_sim_factor == 0.0:
                print('epoch {}, iter {}, total loss: {}, sim_factor: {}, sim_loss: {}, reg_factor: {}, reg_loss: {}'.format(
                    epoch, i + 1, loss.item(), sim_factor, sim_loss.item(), reg_factor, reg_loss.item())
                )
            elif atlas_pair_sim_factor != 0.0 and image_pair_sim_factor == 0.0:
                print('epoch {}, iter {}, total loss: {}, sim_factor: {}, sim_loss: {}, reg_factor: {}, reg_loss: {}, atlas_pair_sim_factor: {}, atlas_pair_sim_loss: {}'.format(
                    epoch, i + 1, loss.item(), sim_factor, sim_loss.item(), reg_factor, reg_loss.item(), atlas_pair_sim_factor, atlas_pair_sim_loss.item())
                )
            elif atlas_pair_sim_factor == 0.0 and image_pair_sim_factor != 0.0:
                print('epoch {}, iter {}, total loss: {}, sim_factor: {}, sim_loss: {}, reg_factor: {}, reg_loss: {}, image_pair_sim_factor: {}, image_pair_sim_loss: {}'.format(
                    epoch, i + 1, loss.item(), sim_factor, sim_loss.item(), reg_factor, reg_loss.item(), image_pair_sim_factor, image_pair_sim_loss.item())
                )
            elif atlas_pair_sim_factor != 0.0 and image_pair_sim_factor != 0.0:
                print('epoch {}, iter {}, total loss: {}, sim_factor: {}, sim_loss: {}, reg_factor: {}, reg_loss: {}, atlas_pair_sim_factor: {}, atlas_pair_sim_loss: {}, image_pair_sim_factor: {}, image_pair_sim_loss: {}'.format(
                    epoch, i + 1, loss.item(), sim_factor, sim_loss.item(), reg_factor, reg_loss.item(), atlas_pair_sim_factor, atlas_pair_sim_loss.item(), image_pair_sim_factor, image_pair_sim_loss.item())
                )

            del svf_warped_atlas_imgs, phi_AB, pos_deform_field, phi_BA, neg_deform_field

        ## Validate to save the best atlas and model parameters
        if epoch % args.save_per_epoch == (args.save_per_epoch - 1):
            with torch.set_grad_enabled(False):
                ## create avg seg
                tmp_img, tmp_seg = 0, 0
                dice_all = 0

                atlas_imgs = atlas_tensor.cuda(args.gpu)
                atlas_segs = atlas_seg_tensor.cuda(args.gpu).float()
                if len(atlas_imgs.shape) != 5:
                    atlas_imgs = atlas_imgs.unsqueeze(0)
                    atlas_seg = atlas_segs.unsqueeze(0)
                for _, (mean_src_imgs, mean_src_segs, _) in enumerate(SVFNet_val_dataloader):
                    mean_src_imgs, mean_src_segs = mean_src_imgs.cuda(args.gpu), mean_src_segs.cuda(args.gpu)


                    _ = train_model(atlas_imgs, mean_src_imgs)

                    mean_neg_deform_field_src = train_model.phi_BA_vectorfield

                    mean_warped_src_segs = compute_warped_image_multiNC(
                        mean_src_segs.float(),
                        train_model.phi_BA_vectorfield,
                        train_model.spacing,
                        0,
                        zero_boundary=True
                    )

                    tmp_seg += mean_warped_src_segs
                mean_atlas_seg_tensor = tmp_seg / len(SVFNet_val_dataloader)

                ## inference
                for _, (inf_src_imgs, inf_src_segs, _) in enumerate(SVFNet_val_dataloader):
                    inf_src_imgs, inf_src_segs = inf_src_imgs.cuda(args.gpu), inf_src_segs.cuda(args.gpu)
                    src_cat_input = torch.cat((atlas_imgs, inf_src_imgs), 1)
                    _ = train_model(atlas_imgs, inf_src_imgs)

                    pos_deform_field = train_model.phi_AB_vectorfield

                    svf_warped_atlas_segs = compute_warped_image_multiNC(
                        mean_atlas_seg_tensor.float(),
                        train_model.phi_AB_vectorfield,
                        train_model.spacing,
                        0,
                        zero_boundary=True
                    )
                    dice_all += (1.0 - get_atlas_seg_loss(inf_src_segs, svf_warped_atlas_segs))

                dice_avg = dice_all / len(SVFNet_val_dataloader)
                print("{} epoch, {} iter, training loss: {:.5f}, val dice: {:.5f}".format(epoch, i + 1, loss.item(), dice_avg))
                writer.add_scalar('validation/dice_avg', dice_avg, global_step=global_step)


                if dice_avg > best_score:
                    best_score = dice_avg.item()
                    print('{} epoch, current highest - Dice: {:.5f}'.format(epoch, dice_avg))
                    writer.add_scalar('validation/highest_dice', dice_avg, global_step=global_step)
                    save_model_path = './ckpoints/' + experiment_name + '/'
                    if not os.path.isdir(save_model_path):
                        os.mkdir(save_model_path)
                    best_state = {'epoch': epoch,
                                  'state_dict': train_model.state_dict(),
                                  'optimizer': optimizer.state_dict(),
                                  'best_score': best_score,
                                  'global_step': global_step
                                  }
                    torch.save(best_state, save_model_path + 'model_best.pth.tar')
                    tmp_img, tmp_seg, JD_denominator = 0, 0, 0
                    for _, (update_src_imgs, update_src_segs, _) in enumerate(SVFNet_atlas_update_dataloader):
                        update_src_imgs, update_src_segs = update_src_imgs.cuda(args.gpu), update_src_segs.cuda(args.gpu)

                        src_cat_input = torch.cat((atlas_imgs, update_src_imgs), 1)
                        _  = train_model(atlas_imgs, update_src_imgs)
                        update_pos_deform_field_src = train_model.phi_AB_vectorfield
                        update_neg_deform_field_src = train_model.phi_BA_vectorfield

                        update_warped_src_imgs = compute_warped_image_multiNC(
                            update_src_imgs.float(),
                            train_model.phi_BA_vectorfield,
                            train_model.spacing,
                            0,
                            zero_boundary=True
                        )

                        update_warped_src_segs = compute_warped_image_multiNC(
                            update_src_segs.float(),
                            train_model.phi_BA_vectorfield,
                            train_model.spacing,
                            0,
                            zero_boundary=True
                        )


                        JD_tensor = torch.from_numpy(jacobian_determinant(update_neg_deform_field_src)).unsqueeze(0).unsqueeze(0).cuda(args.gpu)
                        JD_denominator += JD_tensor

                        tmp_img += (update_warped_src_imgs*JD_tensor)
                        tmp_seg += (update_warped_src_segs*JD_tensor)
                    atlas_tensor = tmp_img / JD_denominator
                    # atlas_tensor.clamp_(min=0.0, max=1.0)
                    atlas_seg = tmp_seg / JD_denominator
                    save_atlas_img_name = save_model_path + 'atlas_svf_img_epoch_' + str(1000+epoch) + '_' + loss_name + '_' + str(best_score) + '.nii.gz'
                    save_atlas_est_name = save_model_path + 'atlas_svf_est_epoch_' + str(1000+epoch) + '_' + loss_name + '_' + str(best_score) + '.nii.gz'
                    save_atlas_prob_name = save_model_path + 'atlas_svf_prob_epoch_' + str(1000+epoch) + '_' + loss_name + '_' + str(best_score) + '.pt'
                    save_updated_atlas(atlas_tensor, atlas_seg, save_atlas_img_name, save_atlas_est_name, save_atlas_prob_name)

            save_model_path = './ckpoints/' + experiment_name + '/'
            if not os.path.isdir(save_model_path):
                os.mkdir(save_model_path)
            current_state = {'epoch': epoch,
                             'state_dict': train_model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'best_score': best_score,
                             'global_step': global_step
                            }
            torch.save(current_state, save_model_path + 'checkpoint.pth.tar')



    writer.close()