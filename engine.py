from typing import Iterable
import torch
import time
from util.misc import NestedTensor
from tqdm import tqdm

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, args = None, logger=None):
    model.train()
    criterion.train()

    pbar = tqdm(total=len(data_loader.dataloader), desc=f"Initial Loss Fused: Pending")
    for i, data in enumerate(data_loader):
        mask = data['label'].squeeze(1).bool()
        samples = NestedTensor(data['image'], mask).to(torch.device(args.device))
        targets = data['label'].to(torch.device(args.device))

        output_fused = model(samples)
        loss = criterion(output_fused, targets.float())

        curTime = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))
        loss_side_total_str = 0
        loss_str = '{:.4f}'.format(loss.item())
        l = optimizer.param_groups[0]['lr']
        logger.info(
            f"time -> {curTime} | Epoch -> {epoch} | image_num -> {data['A_paths']} | loss side total -> {loss_side_total_str} | loss -> {loss_str} | lr -> {l}")

        pbar.set_description(f"Loss: {loss.item():.4f}")
        pbar.update(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    pbar.close()





