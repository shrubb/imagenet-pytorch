import argparse
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

import torchvision

from tensorboardX import SummaryWriter

model_names = sorted(name for name in torchvision.models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision.models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--dataset-root', type=Path,
                    help='path to dataset')
parser.add_argument('--architecture', default='resnet18', choices=model_names,
                    help='Model architecture')
parser.add_argument('--num-workers', default=4, type=int,
                    help='Number of data loading subprocesses. 0 means data will be '
                         'loaded synchronously')
parser.add_argument('--num-epochs', default=90, type=int,
                    help='Number of epochs to run')
parser.add_argument('--batch-size', default=256, type=int,
                    help='Mini-batch size')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                    help='Device to train on')
parser.add_argument('--image-size', default=224, type=int,
                    help='Network input\'s spatial size')

parser.add_argument('--run-name', default=time.ctime(time.time())[4:-8], type=str,
                    help='Experiment name. Tensorboard logs and checkpoints will be put to "./runs/<run-name>" '
                         '(default: date and time formatted as "Dec  9 12:15:30")')
parser.add_argument('--resume', default=None, type=Path,
                    help='Path to a checkpoint to start from (default: start from scratch)')
parser.add_argument('--iteration', default=0, type=int,
                    help='Iteration number to start from (sometimes useful when restarting from a checkpoint)')
parser.add_argument('--seed', default=None, type=int,
                    help='Integer seed for initializing random number generators')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='Only evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='Initialize pre-trained weights from torchvision library')

# Optimizer
parser.add_argument('--optimizer', default='Adam', type=str,
                    help='optimizer type (default: Adam)')
parser.add_argument('--lr', '--learning-rate', default=None, type=float, # 0.1
                    help='Initial learning rate')
parser.add_argument('--momentum', default=None, type=float, # 0.9
                    help='Momentum value')
parser.add_argument('--nesterov', default=None, type=bool, # True
                    help='Use Nesterov momentum?')
parser.add_argument('--weight-decay', '--wd', default=None, type=float, # 1e-4
                    help='Weight decay (default: 1e-4)')

# Distributed (multi-node) training
parser.add_argument('--world-size', default=1, type=int,
                    help='Total number of distributed processes (DO NOT SET)')
parser.add_argument('--rank', default=0, type=int,
                    help='The index of this process within all training processes (DO NOT SET)')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='Master URL for distributed training')
parser.add_argument('--dist-backend', default='nccl', choices=['nccl', 'gloo', 'mpi'],
                    help='Distributed backend')

parser.add_argument('--lr-test', action='store_true',
                    help='Do a learning rate test to prepare for superconvergence runs')

best_top1_accuracy = 0
is_distributed = False

def main():
    global args, best_top1_accuracy, is_distributed
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    is_distributed = args.world_size > 1

    if is_distributed:
        torch.distributed.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size)

    # Create model
    print(f"Will train '{args.architecture}' architecture")
    if args.pretrained:
        print("'--pretrained' is set, so loading pre-trained weights from torchvision")
    model = torchvision.models.__dict__[args.architecture](pretrained=args.pretrained)

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)
        if args.device == 'cuda':
            torch.cuda.set_device(args.rank)

    model = model.to(args.device)

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)

    optimizer_kwargs = {hyperparam: getattr(args, hyperparam) \
        for hyperparam in ('lr', 'weight_decay', 'momentum', 'nesterov') \
        if getattr(args, hyperparam) is not None}
    optimizer = torch.optim.__dict__[args.optimizer](model.parameters(), **optimizer_kwargs)

    # Optionally resume from a checkpoint
    if args.resume:
        if args.resume.is_file():
            print(f"Loading checkpoint '{args.resume}'...")
            start_time = time.time()

            checkpoint = torch.load(args.resume)
            args.iteration = checkpoint['iteration'] * checkpoint['batch_size'] // args.batch_size
            best_top1_accuracy = checkpoint['best_top1_accuracy']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            print(f"Done loading checkpoint. Captured at iteration {checkpoint['iteration']} "
                   "(with our batch size it would be {args.iteration}). Took {time.time() - start_time}")
        else:
            raise FileNotFoundError(f"The requested checkpoint at '{args.resume}' doesn't exist")

    torch.backends.cudnn.benchmark = True

    # Data loading code
    traindir = args.dataset_root / "train"
    valdir = args.dataset_root / "val"
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(args.image_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize,
        ]))

    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    val_dataset = torchvision.datasets.ImageFolder(
        valdir,
        torchvision.transforms.Compose([
            torchvision.transforms.Resize(args.image_size),
            torchvision.transforms.CenterCrop(args.image_size),
            torchvision.transforms.ToTensor(),
            normalize,
        ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    # How many epochs have already been completed?
    args.num_epochs -= args.iteration // len(train_loader)
    print(f"Starting from iteration {args.iteration}, will train for {args.num_epochs} epochs")

    output_directory = Path("runs") / args.run_name
    output_directory.mkdir(parents=True, exist_ok=True)

    checkpoints_directory = output_directory / "checkpoints"
    checkpoints_directory.mkdir(exist_ok=True)
    
    # Initialize tensorboard logger
    tensorboard_writer = SummaryWriter(str(output_directory))

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.num_epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        # Train for one epoch
        train_metrics = train(train_loader, model, criterion, optimizer, tensorboard_writer)
        
        # Evaluate on the validation set
        val_metrics = validate(val_loader, model, criterion)

        # Record metrics to tensorboard
        for tra,val,name in zip(train_metrics, val_metrics, ('Top 1 accuracy', 'Top 5 accuracy', 'Loss')):
            tensorboard_writer.add_scalar(f"Train/{name}", tra, args.iteration)
            tensorboard_writer.add_scalar(f"Validation/{name}", val, args.iteration)
        
        # Save checkpoint
        top1_accuracy = val_metrics[0]
        best_top1_accuracy = max(top1_accuracy, best_top1_accuracy)

        torch.save({
            'iteration': args.iteration,
            'batch_size': args.batch_size,
            'architecture': args.architecture,
            'state_dict': model.state_dict(),
            'best_top1_accuracy': best_top1_accuracy,
            'optimizer' : optimizer.state_dict(),
        }, checkpoints_directory / "last.pth")

        if top1_accuracy > best_top1_accuracy: # this checkpoint is better than all previous ones
            shutil.copy(checkpoints_directory / "last.pth", checkpoints_directory / "best.pth")


def train(train_loader, model, criterion, optimizer, tensorboard_writer):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    total_batches = len(train_loader) if not args.lr_test else 9

    data_loading_start_time = time.time()
    for i, (input, target) in enumerate(train_loader):
        if i > total_batches: break

        # measure data loading time
        data_loading_time = time.time() - data_loading_start_time

        input = input.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        top1_accuracy, top5_accuracy = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), len(input))
        top1.update(top1_accuracy[0], len(input))
        top5.update(top5_accuracy[0], len(input))

        # Compute gradient and do SGD step
        adjust_learning_rate(optimizer, args.lr, args.iteration, len(train_loader))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        total_batch_time = time.time() - data_loading_start_time

        tensorboard_writer.add_scalar('Train/Learning rate',
            next(iter(optimizer.param_groups))['lr'], args.iteration)
        tensorboard_writer.add_scalar('Train/Time/Total batch time', total_batch_time, args.iteration)
        tensorboard_writer.add_scalar('Train/Time/Data loading time', data_loading_time, args.iteration)
        tensorboard_writer.add_scalar('Online batch loss', losses.last_value, args.iteration)
        tensorboard_writer.add_scalar('Online accuracies/Top1', top1.last_value, args.iteration)
        tensorboard_writer.add_scalar('Online accuracies/Top5', top5.last_value, args.iteration)

        args.iteration += 1
        data_loading_start_time = time.time()

    return top1.average, top5.average, losses.average


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            top1_accuracy, top5_accuracy = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(top1_accuracy[0], input.size(0))
            top5.update(top5_accuracy[0], input.size(0))

    return top1.average, top5.average, losses.average


class AverageMeter(object):
    """
        Computes and stores the average and the current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.last_value = val
        self.sum += val * n
        self.count += n
        self.average = self.sum / self.count


def adjust_learning_rate(optimizer, initial_lr, iteration, iterations_per_epoch):
    """
        Defines the learning rate schedule.
    """
    epoch_number = iteration // iterations_per_epoch
    lr = initial_lr * (0.1 ** (epoch_number // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """
        Computes the top-K accuracy for all of the specified values of K.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
