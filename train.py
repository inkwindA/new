import os
import torch
import torch.nn.utils as nn_utils
import argparse
from model import MultiFrameCTDenoiser
import train_dataset
import math
from matplotlib import pyplot
import wandb
wandb.init(project="newCT")
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--dataset_path', type=str, default='dataset\L333')
arg_parser.add_argument('--checkpoint_path', type=str, default='checkpoint')
arg_parser.add_argument('--n_epochs', type=int, default=200)
arg_parser.add_argument('--batch_size', type=int, default=1)
arg_parser.add_argument('--threads', type=int, default=0)
arg_parser.add_argument("--warmup", type=int, default=100)
arg_parser.add_argument("--init_lr", type=float, default=1e-6)
arg_parser.add_argument("--final_lr", type=float, default=1e-6)
arg_parser.add_argument("--cos_lr", type=bool, default=True)
arg_parser.add_argument('--b1', type=float, default=0.99)
arg_parser.add_argument('--b2', type=float, default=0.999)
arg_parser.add_argument('--clip_grad', type=float, default=0,  
                        help='梯度裁剪阈值，设为0或None表示不裁剪')
args = arg_parser.parse_args()
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建checkpoint目录
if not os.path.exists(args.checkpoint_path):
    os.makedirs(args.checkpoint_path)

train_dataset = train_dataset.TrainDataset(args.dataset_path)
train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                num_workers=args.threads,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                )

model = MultiFrameCTDenoiser(num_frames=3, in_channels=1, out_channels=1, base_channels=64).to(device)
print('model parameter size=' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, betas=(args.b1, args.b2))
loss_function = torch.nn.L1Loss().to(device)


for epoch in range(args.n_epochs):
    for i, (ldct, ndct) in enumerate(train_data_loader):
        n_iter = epoch * len(train_dataset) + i + 1
        max_iter = args.n_epochs * len(train_dataset)
        ldct = torch.autograd.Variable(ldct).to(device)
        ndct = torch.autograd.Variable(ndct).to(device)

        optimizer.zero_grad()

        if args.cos_lr:
            lr = (
                math.cos(
                    (n_iter - args.warmup) * (math.pi / (max_iter - args.warmup))
                )
                * 0.5
                + 0.5
            ) * (args.init_lr - args.final_lr) + args.final_lr
        else:
            lr = args.init_lr * (args.final_lr / args.init_lr) ** (
                (n_iter - args.warmup) / (max_iter)
            )

        for p in optimizer.param_groups:
            p["lr"] = lr

        pixel_lamda = 100

        # 直接使用数据集提供的多帧输入
        pred = model(ldct) #1
        pixel_loss = loss_function(pred, ndct)
        loss = pixel_lamda * (pixel_loss)

        loss.backward()


        if args.clip_grad is not None and args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.clip_grad,
                norm_type=2
            )
        optimizer.step()

        print('epoch=%d/%d | step=%d/%d | loss=%f, pixel loss=%f' % (epoch,
                                                                                   args.n_epochs,
                                                                                   i,
                                                                                   len(train_dataset),
                                                                                   loss / 2,
                                                                                   pixel_loss))
        wandb.log({"loss": loss.item()/2, "pixel_loss": pixel_loss.item(), "lr": lr})

    torch.save(model.state_dict(), os.path.join(args.checkpoint_path, 'inknet%d.pth' % epoch))
