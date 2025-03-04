import os
import argparse
import random
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from model.PIDSRNet import PIDSRNet
from utils import create_dataset
from utils.utils import visulize_aop_dop, polar_loss, calculate_stokes
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import logging


os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default='')
parser.add_argument('--test_path', type=str, default='')
parser.add_argument('--output', type=str, default='./results')
parser.add_argument('--patch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--rgb_range', type=int, default=255)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--cuda", action="store_true", default=True, help="use cuda")

args = parser.parse_args()


# Configure logger
def setup_logger(output_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_file = os.path.join(output_dir, 'training_1.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def test(model, test_data_loader, device, output_dir, epoch, rgb_range, logger):
    model.eval()

    best_score = 0

    PID_PSNR, PID_SSIM, PID_AoP_MAE = 0, 0, 0
    PISR_PSNR, PISR_SSIM, PISR_AoP_MAE = 0, 0, 0

    with torch.no_grad():
        for input_img, _, gt_lr, gt_hr in test_data_loader:
            input_img, gt_lr, gt_hr = input_img.to(device), gt_lr.to(device), gt_hr.to(device)

            _, demosaic_out = model(input_img)
            _, sr_out = model(demosaic_out)

            # calculate stokes parameters and aop, dop
            lr_0, lr_45, lr_90, lr_135, lr_S0, lr_S1, lr_S2, lr_aop, lr_dop = calculate_stokes(gt_lr, max_val=rgb_range)
            hr_0, hr_45, hr_90, hr_135, hr_S0, hr_S1, hr_S2, hr_aop, hr_dop = calculate_stokes(gt_hr, max_val=rgb_range)
            demosaic_0, demosaic_45, demosaic_90, demosaic_135, demosaic_S0, demosaic_S1, demosaic_S2, demosaic_aop, demosaic_dop = calculate_stokes(demosaic_out, max_val=rgb_range)
            sr_0, sr_45, sr_90, sr_135, sr_S0, sr_S1, sr_S2, sr_aop, sr_dop = calculate_stokes(sr_out, max_val=rgb_range)

            # calculate PSNR, SSIM, AoP MAE for PID and PISR
            pid_psnr_s0 = psnr(demosaic_S0, lr_S0, data_range=2)
            PID_PSNR += pid_psnr_s0
            pid_ssim_s0 = ssim(demosaic_S0, lr_S0, data_range=2, multichannel=True, channel_axis=-1)
            PID_SSIM += pid_ssim_s0
            pid_aop_mae = np.mean(np.minimum(np.abs(lr_aop - demosaic_aop), np.abs(np.abs(lr_aop - demosaic_aop) - np.pi))) / np.pi * 180
            PID_AoP_MAE += pid_aop_mae

            pisr_psnr_s0 = psnr(sr_S0, hr_S0, data_range=2)
            PISR_PSNR += pisr_psnr_s0
            pisr_ssim_s0 = ssim(sr_S0, hr_S0, data_range=2, multichannel=True, channel_axis=-1)
            PISR_SSIM += pisr_ssim_s0
            pisr_aop_mae = np.mean(np.minimum(np.abs(hr_aop - sr_aop), np.abs(np.abs(hr_aop - sr_aop) - np.pi))) / np.pi * 180
            PISR_AoP_MAE += pisr_aop_mae

        PID_PSNR /= len(test_data_loader)
        PID_SSIM /= len(test_data_loader)
        PID_AoP_MAE /= len(test_data_loader)

        PISR_PSNR /= len(test_data_loader)
        PISR_SSIM /= len(test_data_loader)
        PISR_AoP_MAE /= len(test_data_loader)

        demosaic_score = 0.4 * PID_PSNR - 0.6 * PID_AoP_MAE
        sr_score = 0.4 * PISR_PSNR - 0.6 * PISR_AoP_MAE
        score = 0.5 * demosaic_score + 0.5 * sr_score

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(output_dir, f'best_model.pth'))
            
        logger.info("===> Validation:")
        logger.info("===> PID Results <===")
        logger.info(f"AoP MAE: {PID_AoP_MAE:.4f}, PSNR: {PID_PSNR:.4f}, SSIM: {PID_SSIM:.4f}")
        logger.info("===> PISR Results <===")
        logger.info(f"AoP MAE: {PISR_AoP_MAE:.4f}, PSNR: {PISR_PSNR:.4f}, SSIM: {PISR_SSIM:.4f}")


def train():
    # Create output directory if not exists
    os.makedirs(args.output, exist_ok=True)

    # Setup logger
    logger = setup_logger(args.output)

    logger.info("Arguments:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    torch.backends.cudnn.benchmark = True

    # set random seed
    torch.manual_seed(args.seed)
    seed = args.seed
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info(f"Random Seed: {seed}")
    random.seed(seed)
    torch.manual_seed(seed)

    # set GPU
    cuda = args.cuda
    device = torch.device('cuda' if cuda else 'cpu')

    # load data
    logger.info("===> Loading datasets")
    train_dataset = create_dataset.PIDSRData(args, is_train=True)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    logger.info(f"Training data size: {len(train_dataset)}")
    test_dataset = create_dataset.PIDSRData(args, is_train=False)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    logger.info(f"Test data size: {len(test_dataset)}")

    # load models
    logger.info("===> Building models")
    model = PIDSRNet()
    
    if cuda:
        model = model.to(device)

    logger.info("\nPIDSRNet Architecture:\n" + str(model))

    # set optimizer for both models
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # start training
    logger.info("===> Start Training")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        avg_pid_loss = 0
        avg_pisr_loss = 0

        logger.info(f"Training Epoch [{epoch}] ...")

        with tqdm(total=len(train_data_loader), desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for input_img, gt_llr, gt_lr, gt_hr in train_data_loader:
                input_img = input_img.to(device)
                gt_llr = gt_llr.to(device)
                gt_lr = gt_lr.to(device)
                gt_hr = gt_hr.to(device)

                model.zero_grad()

                # Forward pass for PIDNet
                mid_out1, demosaic_out = model(input_img)

                # Forward pass for PISRNet
                mid_out2, sr_out = model(demosaic_out)

                demosaic_loss = polar_loss(demosaic_out, gt_lr, device) + polar_loss(mid_out1, gt_llr, device)
                sr_loss = polar_loss(sr_out, gt_hr, device) + polar_loss(mid_out2, gt_lr, device) * 0.1
                total_loss = demosaic_loss + sr_loss

                # Backward
                total_loss.backward()

                optimizer.step()

                epoch_loss += total_loss.item()
                avg_pid_loss += demosaic_loss.item()
                avg_pisr_loss += sr_loss.item()

                # Update progress bar
                pbar.set_postfix({'PID Loss': demosaic_loss.item(), 'PISR Loss': sr_loss.item()})
                pbar.update(1)

        # Scheduler step
        scheduler.step()

        avg_epoch_loss = epoch_loss / len(train_data_loader)
        avg_pid_loss /= len(train_data_loader)
        avg_pisr_loss /= len(train_data_loader)

        logger.info(f"===> Epoch {epoch+1} Complete: Avg. Loss: {avg_epoch_loss:.4f}, PID Loss: {avg_pid_loss:.4f}, PISR Loss: {avg_pisr_loss:.4f}")

        # Validation
        test(model, test_data_loader, device, args.output, epoch, args.rgb_range, logger)


if __name__ == '__main__':
    train()
