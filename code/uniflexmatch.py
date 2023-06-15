import yaml
import torch
import pprint
import logging
import argparse
import os, math
import numpy as np
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from util.classes import CLASSES
from dataset.semi import SemiDataset
from util.ohem import ProbOhemCrossEntropy2d
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.utils import init_log, AverageMeter, intersectionAndUnion

class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value
    
def evaluate(model, loader, mode, cfg):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with torch.no_grad():
        for img, mask, id in loader:
            img = img.cuda()
            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, 19, h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)
                pred = final.argmax(dim=1)
            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                pred = model(img).argmax(dim=1)
            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)
    return mIOU, iou_class

parser = argparse.ArgumentParser(
    description='Revisiting Weak-to-Strong Consistency \
        in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', 
                    default="/root/UniMatch/configs/pascal.yaml",
                    type=str, required=False)

parser.add_argument('--labeled-id-path', 
                    default="splits/pascal/1_4/labeled.txt",
                    type=str, required=False)

parser.add_argument('--unlabeled-id-path', 
                    default="splits/pascal/1_4/unlabeled.txt",
                    type=str, required=False)

parser.add_argument('--save-path', 
                    default="/root/UniMatch/exp/pascal/uniflexmatch/r50/1_4_1",
                    # default="/root/UniMatch/exp/pascal/unimatch/r50/1_4",
                    type=str, 
                    required=False)

parser.add_argument('--temperature_param', 
                    default=0.5,
                    type=float, required=False)

parser.add_argument('--cutoff_param', 
                    default=0.95,
                    type=float, required=False)

parser.add_argument('--local_rank', 
                    default=0, 
                    type=int)
parser.add_argument('--port', default=None, type=int)

def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    # temperature params function
    temperature_fn = Get_Scalar(args.temperature_param)  
    # confidence cutoff function
    cutoff_fn = Get_Scalar(args.cutoff_param)  
    categories_accurity = torch.zeros(cfg['nclass'], )
    preded_labels_nums = {cls_idx : 10 for cls_idx in range(cfg['nclass'])}
    # rank, world_size = setup_distributed(port=args.port)
    rank = 0
    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': 1}
        logger.info('{}'.format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True
    model = DeepLabV3Plus(cfg)
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    
    local_rank = 0
    model.cuda()
    
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    trainloader_l = DataLoader(trainset_l, shuffle=True, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=4, drop_last=True, sampler=None)
    trainloader_u = DataLoader(trainset_u, shuffle=True,batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=4, drop_last=True, sampler=None)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4,
                           drop_last=False, sampler=None)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    epoch = -1

    train_iter = 0
    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
            epoch, optimizer.param_groups[0]['lr'], previous_best))
                
        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()

        loader = zip(trainloader_l, trainloader_u, trainloader_u)
        for i, ((img_x, mask_x), 
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2), 
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):
            for cls_idx in range(cfg["nclass"]):
                if cls_idx == 0:
                    # background.
                    categories_accurity[cls_idx] = cfg["conf_thresh"]
                    continue
                categories_accurity[cls_idx] = math.log(preded_labels_nums[cls_idx]) / \
                    math.log(max(preded_labels_nums.values()))            
                categories_accurity[cls_idx] = \
                    min(categories_accurity[cls_idx], cfg["conf_thresh"])
                categories_accurity[cls_idx] = \
                    max(categories_accurity[cls_idx], 0.65)                    
            train_iter += 1            
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = \
                img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            with torch.no_grad():
                model.eval()
                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)
            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
            preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]
            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]
            
            loss_x = criterion_l(pred_x, mask_x)
            # T = temperature_fn(train_iter)
            p_cutoff = cutoff_fn(train_iter)
            pixels_u_s1 = conf_u_w_cutmixed1.ge(p_cutoff)
            valid_conf_u_s1 = torch.zeros_like(conf_u_w_cutmixed1)
            for cls_id in range(cfg["nclass"]):
                per_cls_mask = mask_u_w_cutmixed1 == cls_id
                if cls_id != 0:
                    num_of_selected_pixels = (pixels_u_s1 * per_cls_mask).sum().item()
                    preded_labels_nums[cls_id] += num_of_selected_pixels
                valid_conf_u_s1 += per_cls_mask * conf_u_w_cutmixed1.ge(
                    p_cutoff * (categories_accurity[cls_id] / (1.9 - categories_accurity[cls_id]))).float()
                   
            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * (valid_conf_u_s1 * (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

            pixels_u_s2 = conf_u_w_cutmixed2.ge(p_cutoff)
            valid_conf_u_s2 = torch.zeros_like(conf_u_w_cutmixed2)
            for cls_id in range(cfg["nclass"]):
                per_cls_mask = mask_u_w_cutmixed2 == cls_id
                if cls_id != 0:
                    num_of_selected_pixels = (pixels_u_s2 * per_cls_mask).sum().item()
                    preded_labels_nums[cls_id] += num_of_selected_pixels
                valid_conf_u_s2 += per_cls_mask * conf_u_w_cutmixed2.ge(
                    p_cutoff * (categories_accurity[cls_id] / (1.9 - categories_accurity[cls_id]))).float()
                   
            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * (valid_conf_u_s2 * (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255))
            loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()

            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())
            
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
            
            if i % 20 == 0 and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg,
                                            total_loss_w_fp.avg, total_mask_ratio.avg))

        if epoch % 2 != 0:
            continue
        logger.info("categories_accurity: {}".format(categories_accurity.tolist()))
        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        if rank == 0:
            logger.info("******************** {} ***************************".format(epoch))
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}'.format(eval_mode, mIoU))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))

if __name__ == '__main__':
    main()
