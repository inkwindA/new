import os
import psnr
import torch
import shutil
import argparse
from model import MultiFrameCTDenoiser
import test_dataset
import matplotlib.pyplot

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--dataset_path', type=str, default='dataset/L506')
arg_parser.add_argument('--checkpoint_path', type=str, default='checkpoint')
# arg_parser.add_argument('--checkpoint', type=int, default=140)
arg_parser.add_argument('--save_path', type=str, default='save/fig')
arg_parser.add_argument('--start_checkpoint', type=int, default=0)
arg_parser.add_argument('--end_checkpoint', type=int, default=20)
arg_parser.add_argument('--precision_path', type=str, default='save/precision')
arg_parser.add_argument('--batch_size', type=int, default=1)

arg_parser.add_argument("--threads", type=int, default=0)
args = arg_parser.parse_args(args=[])
print(args)

trunc_min = -160.0
trunc_max = 240.0


def denormalize(image): 
    norm_range_min = -1024.0
    norm_range_max = 3072.0
    image = image * (norm_range_max - norm_range_min) + norm_range_min

    return image


def trunc(mat):
    mat[mat <= trunc_min] = trunc_min
    mat[mat >= trunc_max] = trunc_max

    return mat


if os.path.exists(args.save_path):
    shutil.rmtree(args.save_path)
os.makedirs(args.save_path)


def save_fig(index, ldct, ndct, pred, ldct_result, pred_result):
    ldct, ndct, pred = ldct.numpy(), ndct.numpy(), pred.numpy()
    fig, ax = matplotlib.pyplot.subplots(1, 3, figsize=(30, 10))
    ax[0].imshow(ldct, cmap=matplotlib.pyplot.cm.gray, vmin=trunc_min, vmax=trunc_max)
    ax[0].set_title('Ldct', fontsize=30)
    ax[0].set_xlabel("PSNR=%04f\nSSIM=%04f\nRMSE=%04f" % (ldct_result[0],
                                                          ldct_result[1],
                                                          ldct_result[2]),
                     fontsize=20)

    ax[1].imshow(pred, cmap=matplotlib.pyplot.cm.gray, vmin=trunc_min, vmax=trunc_max)
    ax[1].set_title('Pred', fontsize=30)
    ax[1].set_xlabel("PSNR=%04f\nSSIM=%04f\nRMSE=%04f" % (pred_result[0],
                                                          pred_result[1],
                                                          pred_result[2]),
                     fontsize=20)

    ax[2].imshow(ndct, cmap=matplotlib.pyplot.cm.gray, vmin=trunc_min, vmax=trunc_max)
    ax[2].set_title('Ndct', fontsize=30)

    fig.savefig(os.path.join(args.save_path, 'result%d.png' % index))
    matplotlib.pyplot.close()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_dataset = test_dataset.TestDataset(args.dataset_path)
test_data_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    num_workers=args.threads,
    batch_size=args.batch_size,
    shuffle=False)

model = MultiFrameCTDenoiser(num_frames=3, in_channels=1, out_channels=1, base_channels=64)
model.to(device)


with torch.no_grad():
     for checkpoint in range(args.start_checkpoint, args.end_checkpoint + 1):
            model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, 'inknet%d.pth' % checkpoint), map_location='cuda'))
            ldct_psnr, ldct_ssim, ldct_rmse = 0, 0, 0
            pred_psnr, pred_ssim, pred_rmse = 0, 0, 0
            for i, (ndct, ldct) in enumerate(test_data_loader):
                ldct = torch.autograd.Variable(ldct).cuda()
                ndct = torch.autograd.Variable(ndct).cuda()

                pred = model(ldct)

                # 只取当前帧（中间帧）进行可视化
                current_ldct = ldct[:, 1:2, :, :]  # 取中间帧
                
                current_ldct = trunc(denormalize(current_ldct.view(current_ldct.shape[-2], current_ldct.shape[-1]).cpu()))
                ndct = trunc(denormalize(ndct.view(ndct.shape[-2], ndct.shape[-1]).cpu()))
                pred = trunc(denormalize(pred.view(pred.shape[-2], pred.shape[-1]).cpu()))

                ldct_result, pred_result = psnr.compute_measure(current_ldct, ndct, pred, trunc_max - trunc_min)
                ldct_psnr += ldct_result[0]
                ldct_ssim += ldct_result[1]
                ldct_rmse += ldct_result[2]
                pred_psnr += pred_result[0]
                pred_ssim += pred_result[1]
                pred_rmse += pred_result[2]

                print('step=%d/%d | ' \
                    'ldct(psnr=%f, ssim=%f, rmse=%f) | ' \
                    'pred(psnr=%f, ssim=%f, rmse=%f)' % (i,
                                                        len(test_dataset),
                                                        ldct_result[0],
                                                        ldct_result[1],
                                                        ldct_result[2],
                                                        pred_result[0],
                                                        pred_result[1],
                                                        pred_result[2]))

                # save_fig(i, ldct, ndct, pred, ldct_result, pred_result)

            print('Ldct:\nPSNR=%04f\nSSIM=%04f\nRMSE=%04f' % (ldct_psnr / len(test_dataset),
                                                            ldct_ssim / len(test_dataset),
                                                            ldct_rmse / len(test_dataset)))
            print('Pred:\nPSNR=%04f\nSSIM=%04f\nRMSE=%04f' % (pred_psnr / len(test_dataset),
                                                            pred_ssim / len(test_dataset),
                                                            pred_rmse / len(test_dataset)))

            if checkpoint == args.start_checkpoint:
                with open(os.path.join(args.precision_path, 'precision.txt'), 'w') as file:
                    print('--- %04f %04f %04f\n    PSNR      SSIM     RMSE' % (ldct_psnr / len(test_dataset),
                                                                                ldct_ssim / len(test_dataset),
                                                                                ldct_rmse / len(test_dataset))
                                                                                                    ,
                        file=file
                        )
            with open(os.path.join(args.precision_path, 'precision.txt'), 'a') as file:
                print('%03d %04f %04f %04f' % (checkpoint,
                                                pred_psnr / len(test_dataset),
                                                pred_ssim / len(test_dataset),
                                                pred_rmse / len(test_dataset)
                                                ),
                    file=file
                    )
