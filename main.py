import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import datetime
import random
import time
from pathlib import Path
import numpy as np
import torch
import util.misc as utils
from engine import train_one_epoch
from models import build_model
from datasets import create_dataset
import cv2
from eval.evaluate import eval
from util.logger import get_logger
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser('CrackSCF', add_help=False)

    # args for Train
    parser.add_argument('--dataset_path', default="../data/CFD", help='path to images')
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--lr_drop', default=30, type=int)
    parser.add_argument('--weight_decay', default=5e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--BCELoss_ratio', default=0.75, type=float)
    parser.add_argument('--DiceLoss_ratio', default=0.25, type=float)
    parser.add_argument('--batch_size_train', type=int, default=1, help='train input batch size')
    parser.add_argument('--batch_size_test', type=int, default=1, help='test input batch size')
    parser.add_argument('--batch_size_val', type=int, default=1, help='val input batch size')

    # args for MFE
    parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--spm_on', type=bool, default=True, help='whether use sp in Resnet')

    # args for LDE
    parser.add_argument('--num_queries', default=1024, type=int, help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float, help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=5, type=int, help='number of feature levels')
    parser.add_argument('--enc_layers', default=1, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=1, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads inside the transformer's attentions")

    # args for dataset
    parser.add_argument('--output_dir', default='./checkpoints/weights', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dataset_mode', type=str, default='crack')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, takes them randomly')
    parser.add_argument('--num_threads', default=1, type=int, help='threads for loading data')
    parser.add_argument('--phase', type=str, default='train', help='train, val, etc')
    parser.add_argument('--load_width', type=int, default=384, help='load image width')
    parser.add_argument('--load_height', type=int, default=384, help='load image height')

    return parser

def main(args):

    checkpoints_path = "./checkpoints"
    curTime = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))
    dataset_name = (args.dataset_path).split('/')[-1]
    process_floder_path = os.path.join(checkpoints_path, curTime + '_Dataset->' + dataset_name)
    if not os.path.exists(process_floder_path):
        os.makedirs(process_floder_path)
    else:
        print("create process floder error!")

    log_train = get_logger(process_floder_path, 'train')
    log_val = get_logger(process_floder_path, 'val')
    log_eval = get_logger(process_floder_path, 'eval')

    log_train.info("args -> " + str(args))
    log_train.info("args: n_offset_points -> " + str(args.dec_n_points))
    log_train.info("args: num_queries -> " + str(args.num_queries))
    log_train.info("args: backbone -> " + str(args.backbone))
    log_train.info("args: dataset -> " + str(args.dataset_path))
    log_train.info("args: enc_dec_layers -> " + str(args.enc_layers))
    log_train.info("args: BCELoss_ratio -> " + str(args.BCELoss_ratio))
    log_train.info("args: DiceLoss_ratio -> " + str(args.DiceLoss_ratio))

    print("args: BCELoss_ratio -> " + str(args.BCELoss_ratio))
    print("args: DiceLoss_ratio -> " + str(args.DiceLoss_ratio))

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model(args)
    model.to(device)

    model_without_ddp = model

    args.batch_size = args.batch_size_train
    train_dataLoader = create_dataset(args)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(train_dataLoader)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    log_train.info('The number of training images = %d' % dataset_size)


    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()],
            "lr": args.lr,
        },
    ]
    if args.sgd:
        print('use SGD!')
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        print('use AdamW!')
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    output_dir = Path(args.output_dir)

    print("Start process!")
    log_train.info("Start process!")
    start_time = time.time()
    max_F1 = 0
    max_Metrics = {'epoch': 0, 'mIoU': 0, 'ODS': 0, 'OIS': 0, 'F1': 0, 'Precision': 0, 'Recall': 0}

    for epoch in range(args.start_epoch, args.epochs):
        args.phase = 'train'
        print("---------------------------------------------------------------------------------------")
        print("training epoch start -> ", epoch)

        train_one_epoch(
            model, criterion, train_dataLoader, optimizer, epoch, args, log_train)

        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        print("training epoch finish -> ", epoch)
        print("---------------------------------------------------------------------------------------")

        print("valing epoch start -> ", epoch)
        results_path = curTime + '_Dataset->' + dataset_name + '_BCERatio->' + str(args.BCELoss_ratio) + '_DiceRatio->' + str(args.DiceLoss_ratio) + '_layerNum->' + str(args.enc_layers)
        save_root = f'./results/{results_path}/results_' + str(epoch)
        args.phase = 'val'
        args.batch_size = args.batch_size_val
        val_dl = create_dataset(args)
        pbar = tqdm(total=len(val_dl), desc=f"Initial Loss: Pending")

        if not os.path.isdir(save_root):
            os.makedirs(save_root)
        with torch.no_grad():
            for batch_idx, (data) in enumerate(val_dl):
                x = data["image"]
                target = data["label"]
                if device != 'cpu':
                    x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                out = model(x)

                loss = criterion(out, target.float())

                target = target[0, 0, ...].cpu().numpy()
                out = out[0, 0, ...].cpu().numpy()
                root_name = data["A_paths"][0].split("/")[-1][0:-4]

                target = 255 * (target / np.max(target))
                out = 255 * (out / np.max(out))

                # out[out >= 0.3] = 255
                # out[out < 0.3] = 0

                log_val.info('----------------------------------------------------------------------------------------------')
                log_val.info("loss -> " + str(loss))
                log_val.info(str(os.path.join(save_root, "{}_lab.png".format(root_name))))
                log_val.info(str(os.path.join(save_root, "{}_pre.png".format(root_name))))
                log_val.info('----------------------------------------------------------------------------------------------')
                cv2.imwrite(os.path.join(save_root, "{}_lab.png".format(root_name)), target)
                cv2.imwrite(os.path.join(save_root, "{}_pre.png".format(root_name)), out)
                pbar.set_description(f"Loss: {loss.item():.4f}")
                pbar.update(1)

        pbar.close()
        log_val.info("model -> " + str(epoch) + " val finish!")
        log_val.info('----------------------------------------------------------------------------------------------')
        print("validating epoch finish -> ", epoch)

        print("---------------------------------------------------------------------------------------")
        print("evalauting epoch start -> ", epoch)
        metrics = eval(log_eval, save_root, epoch)
        for key, value in metrics.items():
            print(str(key) + ' -> ' + str(value))
        if(max_F1 < metrics['F1']):
            max_Metrics = metrics
            max_F1 = metrics['F1']
            checkpoint_paths = [output_dir / f'checkpoint_best.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            log_train.info("\nupdate and save best model -> " + str(epoch))
            print("\nupdate and save best model -> ", epoch)

        print("evalauting epoch finish -> ", epoch)
        print('\nmax_F1 -> ' + str(max_Metrics['F1']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))
        print("---------------------------------------------------------------------------------------")

        log_eval.info("evalauting epoch finish -> " + str(epoch))
        log_eval.info('\nmax_F1 -> ' + str(max_Metrics['F1']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))
        log_eval.info("---------------------------------------------------------------------------------------")

    for key, value in max_Metrics.items():
        log_eval.info(str(key) + ' -> ' + str(value))
    log_eval.info('\nmax_F1 -> ' + str(max_Metrics['F1']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Process time {}'.format(total_time_str))
    log_train.info('Process time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CrackSCF', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
