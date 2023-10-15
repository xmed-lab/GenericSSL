import os
import sys
import logging
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', type=str, default='synapse')
parser.add_argument('--exp', type=str, default='diffusion')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-sl', '--split_labeled', type=str, default='labeled_20p')
parser.add_argument('-su', '--split_unlabeled', type=str, default='unlabeled_80p')
parser.add_argument('-se', '--split_eval', type=str, default='eval')
parser.add_argument('-m', '--mixed_precision', action='store_true', default=True) # <--
parser.add_argument('-ep', '--max_epoch', type=int, default=300)
parser.add_argument('--cps_loss', type=str, default='w_ce+dice')
parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-w', '--mu', type=float, default=2.0)
parser.add_argument('-s', '--ema_w', type=float, default=0.99)
parser.add_argument('-r', '--mu_rampup', action='store_true', default=True) # <--
parser.add_argument('-cr', '--rampup_epoch', type=float, default=None) # 100
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from DiffVNet.diff_vnet_2d import DiffVNet
from utils import EMA, maybe_mkdir, get_lr, fetch_data, GaussianSmoothing, seed_worker, poly_lr, print_func, sigmoid_rampup
from utils.loss import DC_and_CE_loss, RobustCrossEntropyLoss, SoftDiceLoss
from data.data_loaders import DatasetAllTasks
from utils import config
from utils.metrics import mean_iou
config = config.Config(args.task+"2d")
from data.StrongAug_2d import get_StrongAug, ToTensor, CenterCrop


def get_current_mu(epoch):
    if args.mu_rampup:
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        if args.rampup_epoch is None:
            args.rampup_epoch = args.max_epoch
        return args.mu * sigmoid_rampup(epoch, args.rampup_epoch)
    else:
        return args.mu


def make_loss_function(name, weight=None):
    if name == 'ce':
        return RobustCrossEntropyLoss()
    elif name == 'wce':
        return RobustCrossEntropyLoss(weight=weight)
    elif name == 'ce+dice':
        return DC_and_CE_loss()
    elif name == 'wce+dice':
        return DC_and_CE_loss(w_ce=weight)
    elif name == 'w_ce+dice':
        return DC_and_CE_loss(w_dc=weight, w_ce=weight)
    else:
        raise ValueError(name)

def make_loader(split, dst_cls=DatasetAllTasks, repeat=None, is_training=True, unlabeled=False, transforms_tr=None, transforms_val=None):
    if is_training:
        dst = dst_cls(
            split=split,
            repeat=repeat,
            unlabeled=unlabeled,
            transform=transforms_tr,
            task=args.task,
            num_cls=config.num_cls,
            is_2d=True
        )
        return DataLoader(
            dst,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            drop_last=True
        )
    else:
        dst = dst_cls(
            split=split,
            is_val=True,
            task=args.task,
            num_cls=config.num_cls,
            transform=transforms_val,
            is_2d=True
        )
        return DataLoader(dst, pin_memory=True, num_workers=1, shuffle=False)


def make_model_all():
    model = DiffVNet(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        normalization='batchnorm',
        has_dropout=True
    ).cuda()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True
    )

    return model, optimizer




class Difficulty:
    def __init__(self, num_cls, accumulate_iters=20):
        self.last_dice = torch.zeros(num_cls).float().cuda() + 1e-8
        self.dice_func = SoftDiceLoss(smooth=1e-8, do_bg=True)
        self.cls_learn = torch.zeros(num_cls).float().cuda()
        self.cls_unlearn = torch.zeros(num_cls).float().cuda()
        self.num_cls = num_cls
        self.dice_weight = torch.ones(num_cls).float().cuda()
        self.accumulate_iters = accumulate_iters

    def init_weights(self):
        weights = np.ones(self.num_cls) * self.num_cls
        self.weights = torch.FloatTensor(weights).cuda()
        return weights

    def cal_weights(self, pred,  label):
        x_onehot = torch.zeros(pred.shape).cuda()
        output = torch.argmax(pred, dim=1, keepdim=True).long()
        x_onehot.scatter_(1, output, 1)
        y_onehot = torch.zeros(pred.shape).cuda()
        y_onehot.scatter_(1, label, 1)
        cur_dice = self.dice_func(x_onehot, y_onehot, is_training=False)
        delta_dice = cur_dice - self.last_dice
        cur_cls_learn = torch.where(delta_dice>0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)
        cur_cls_unlearn = torch.where(delta_dice<=0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)

        self.last_dice = cur_dice

        self.cls_learn = EMA(cur_cls_learn, self.cls_learn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        self.cls_unlearn = EMA(cur_cls_unlearn, self.cls_unlearn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        cur_diff = (self.cls_unlearn + 1e-8) / (self.cls_learn + 1e-8)

        cur_diff = torch.pow(cur_diff, 1/5)

        self.dice_weight = EMA(1. - cur_dice, self.dice_weight, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        weights = cur_diff * self.dice_weight
        return weights * self.num_cls






if __name__ == '__main__':
    import random
    SEED=args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # make logger file
    snapshot_path = f'./logs/{args.exp}/'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))
    vis_path = os.path.join(snapshot_path, 'vis')
    maybe_mkdir(vis_path)

    # make logger
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S', force=True
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    transforms_train_labeled = get_StrongAug(config.patch_size, 3, 0.7)
    transforms_train_unlabeled = get_StrongAug(config.patch_size, 3, 0.7)
    transforms_val = transforms.Compose([
        CenterCrop(config.patch_size),
        ToTensor()
    ])


    unlabeled_loader = make_loader(args.split_unlabeled, unlabeled=True, transforms_tr=transforms_train_unlabeled)
    labeled_loader = make_loader(args.split_labeled, repeat=len(unlabeled_loader.dataset), transforms_tr=transforms_train_labeled)
    eval_loader = make_loader(args.split_eval, is_training=False, transforms_val=transforms_val)



    logging.info(f'{len(labeled_loader)} itertations per epoch (labeled)')
    logging.info(f'{len(unlabeled_loader)} itertations per epoch (unlabeled)')

    # make model, optimizer, and lr scheduler
    model, optimizer = make_model_all()

    diff = Difficulty(config.num_cls, accumulate_iters=50)
    deno_loss  = make_loss_function(args.sup_loss)
    sup_loss  = make_loss_function(args.sup_loss)
    unsup_loss  = make_loss_function(args.unsup_loss)


    if args.mixed_precision:
        amp_grad_scaler = GradScaler()

    mu = get_current_mu(0)
    best_eval = 0.0
    best_epoch = 0
    for epoch_num in range(args.max_epoch + 1):
        loss_list = []
        loss_sup_list = []
        loss_diff_list = []
        loss_unsup_list = []

        model.train()
        for batch_l, batch_u in tqdm(zip(labeled_loader, unlabeled_loader)):

            for D_theta_name, D_theta_params in model.decoder_theta.named_parameters():
                if D_theta_name in model.denoise_model.decoder.state_dict().keys():
                    D_xi_params = model.denoise_model.decoder.state_dict()[D_theta_name]
                    D_psi_params = model.decoder_psi.state_dict()[D_theta_name]
                    if D_theta_params.shape == D_xi_params.shape:
                        D_theta_params.data = args.ema_w * D_theta_params.data + (1 - args.ema_w) * (D_xi_params.data + D_psi_params.data) / 2.0


            optimizer.zero_grad()
            image_l, label_l = fetch_data(batch_l)
            label_l = label_l.long()
            image_u = fetch_data(batch_u, labeled=False)

            if args.mixed_precision:
                with autocast():
                    shp = (config.batch_size, config.num_cls)+config.patch_size

                    label_l_onehot = F.one_hot(label_l.squeeze(1), num_classes=config.num_cls).float()
                    label_l_onehot = label_l_onehot.permute(0, 3, 1, 2)
                    x_start = label_l_onehot * 2 - 1
                    x_t, t, noise = model(x=x_start, pred_type="q_sample")

                    p_l_xi = model(x=x_t, step=t, image=image_l, pred_type="D_xi_l")
                    p_l_psi = model(image=image_l, pred_type="D_psi_l")

                    L_deno = deno_loss(p_l_xi, label_l)

                    weight_diff = diff.cal_weights(p_l_xi.detach(), label_l)
                    sup_loss.update_weight(weight_diff)
                    L_diff = sup_loss(p_l_psi, label_l)


                    with torch.no_grad():
                        p_u_xi = model(image_u, pred_type="ddim_sample")
                        p_u_psi = model(image_u, pred_type="D_psi_l")
                        smoothing = GaussianSmoothing(config.num_cls, 3, 1, dim=2)
                        p_u_xi = smoothing(F.gumbel_softmax(p_u_xi, dim=1))
                        p_u_psi = F.softmax(p_u_psi, dim=1)
                        pseudo_label = torch.argmax(p_u_xi + p_u_psi, dim=1, keepdim=True)

                    p_u_theta = model(image=image_u, pred_type="D_theta_u")
                    L_u = unsup_loss(p_u_theta, pseudo_label.detach())

                    loss = L_deno + L_diff + mu * L_u

                # backward passes should not be under autocast.
                amp_grad_scaler.scale(loss).backward()
                amp_grad_scaler.step(optimizer)
                amp_grad_scaler.update()

            else:
                raise NotImplementedError

            loss_list.append(loss.item())
            loss_sup_list.append(L_deno.item())
            loss_diff_list.append(L_diff.item())
            loss_unsup_list.append(L_u.item())


        writer.add_scalar('lr', get_lr(optimizer), epoch_num)
        writer.add_scalar('loss/loss', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/deno', np.mean(loss_sup_list), epoch_num)
        writer.add_scalar('loss/unsup', np.mean(loss_unsup_list), epoch_num)
        writer.add_scalar('loss/diff', np.mean(loss_diff_list), epoch_num)
        writer.add_scalars('class_weights', dict(zip([str(i) for i in range(config.num_cls)] ,print_func(weight_diff))), epoch_num)

        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)} | lr : {get_lr(optimizer)} | mu : {mu}')
        logging.info(f"     diff_w: {print_func(weight_diff)}")
        optimizer.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)

        mu = get_current_mu(epoch_num)



        # =======================================================================================
        # Validation
        # =======================================================================================
        if epoch_num % 1 == 0:
            dice_list = [[] for _ in range(config.num_cls-1)]
            model.eval()
            dice_func = SoftDiceLoss(smooth=1e-8, do_bg=False)
            for batch in tqdm(eval_loader):
                with torch.no_grad():
                    image, gt = fetch_data(batch)
                    p_u_theta = model(image, pred_type="D_theta_u")
                    del image

                    shp = (p_u_theta.shape[0], config.num_cls) + p_u_theta.shape[2:]
                    gt = gt.long()

                    y_onehot = F.one_hot(gt, num_classes=config.num_cls).long()
                    y_onehot = y_onehot.squeeze(1).permute(0, 3, 1, 2)

                    p_u_theta = torch.argmax(p_u_theta, dim=1, keepdim=True).long()
                    x_onehot = F.one_hot(p_u_theta, num_classes=config.num_cls).long()
                    x_onehot = x_onehot.squeeze(1).permute(0, 3, 1, 2)



                    dice = dice_func(x_onehot, y_onehot, is_training=False)
                    dice = dice.data.cpu().numpy()
                    for i, d in enumerate(dice):
                        dice_list[i].append(d)

            dice_mean = []
            for dice in dice_list:
                dice_mean.append(np.mean(dice))
            writer.add_scalar('val_dice', np.mean(dice_mean), epoch_num)
            logging.info(f'evaluation epoch {epoch_num}, dice: {np.mean(dice_mean)}, {dice_mean}')
            if np.mean(dice_mean) > best_eval:
                best_eval = np.mean(dice_mean)
                best_epoch = epoch_num
                save_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')
                torch.save({
                    'state_dict': model.state_dict(),
                }, save_path)
                logging.info(f'saving best model to {save_path}')
            logging.info(f'\t best eval dice is {best_eval} in epoch {best_epoch}')
            if epoch_num - best_epoch == config.early_stop_patience:
                logging.info(f'Early stop.')
                break

    writer.close()
