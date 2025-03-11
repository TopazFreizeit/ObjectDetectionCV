import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import os
import numpy as np
import yaml
import random
from tqdm import tqdm
from model.ssd import SSD
import torchvision
from dataset.voc import VOCDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.utils as utils

from torch.utils.tensorboard import SummaryWriter
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'Using CUDA device: {torch.cuda.get_device_name(0)}')
else:
    device = torch.device('cpu')
    print('CUDA not available, using CPU')

def collate_function(data):
    return tuple(zip(*data))

def train(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    dataset_config = config['dataset_params']
    train_config = config['train_params']

    log_dir = os.path.join('runs', train_config['task_name'], datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    voc = VOCDataset('train',
                     im_sets=dataset_config['train_im_sets'],
                     im_size=dataset_config['im_size'])
    train_dataset = DataLoader(voc,
                               batch_size=train_config['batch_size'],
                               shuffle=True,
                               collate_fn=collate_function,
                               num_workers=4,  
                               pin_memory=True)

    model = SSD(config=config['model_params'],
                num_classes=dataset_config['num_classes'])
    model = model.to(device)
    model.train()
    ckpt_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
    if os.path.exists(ckpt_path):
        print("Found checkpoint, attempting to load...")
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            print("Checkpoint loaded successfully")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Training from scratch instead")

    optimizer = optimizer = torch.optim.Adam(params=model.parameters(), lr=train_config['lr'], weight_decay=5E-4)
    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    steps = 0
    for i in range(num_epochs):
        ssd_classification_losses = []
        ssd_localization_losses = []
        for idx, (ims, targets, _) in enumerate(tqdm(train_dataset)):
            for target in targets:
                target['boxes'] = target['bboxes'].float().to(device)
                del target['bboxes']
                target['labels'] = target['labels'].long().to(device)
            images = torch.stack([im.float().to(device) for im in ims], dim=0)
            batch_losses, _ = model(images, targets)
            loss = batch_losses['classification']
            loss += batch_losses['bbox_regression']

            ssd_classification_losses.append(batch_losses['classification'].item())
            ssd_localization_losses.append(batch_losses['bbox_regression'].item())

            writer.add_scalar('Loss/classification', batch_losses['classification'].item(), steps)
            writer.add_scalar('Loss/localization', batch_losses['bbox_regression'].item(), steps)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], steps)

            loss = loss / acc_steps
            loss.backward()

            if (idx + 1) % acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                optimizer.zero_grad()
            if steps % train_config['log_steps'] == 0:
                loss_output = ''
                loss_output += 'SSD Classification Loss : {:.4f}'.format(np.mean(ssd_classification_losses))
                loss_output += ' | SSD Localization Loss : {:.4f}'.format(np.mean(ssd_localization_losses))
                print(loss_output)
            if torch.isnan(loss) or torch.isnan(batch_losses['classification']) or torch.isnan(batch_losses['bbox_regression']):
                print(f"Classification loss: {batch_losses['classification']}")
                print(f"Localization loss: {batch_losses['bbox_regression']}")
                print("NaN detected in losses")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"{name} grad stats: mean={param.grad.mean()}, std={param.grad.std()}")
                exit(0)
            steps += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        optimizer.zero_grad()
        print('Finished epoch {}'.format(i+1))
        loss_output = ''
        loss_output += 'SSD Classification Loss : {:.4f}'.format(np.mean(ssd_classification_losses))
        loss_output += ' | SSD Localization Loss : {:.4f}'.format(np.mean(ssd_localization_losses))
        print(loss_output)
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                         train_config['ckpt_name']))
    print('Done Training...')
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ssd training')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    parser.add_argument('--ckpt_dir', dest='ckpt_dir',
                        default='checkpoints', type=str)
    args = parser.parse_args()
    train(args)
