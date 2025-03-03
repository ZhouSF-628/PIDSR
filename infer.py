import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
from model.PIDSRNet import PIDSRNet as PIDNet
from model.PIDSRNet import PIDSRNet as PISRNet
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from utils.utils import get_img_from_tensor, visulize_aop_dop, color_polar_mosaic, pad_image

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()

parser.add_argument('--infer_path', type=str, default='./demo/real')
parser.add_argument('--pid_pth_path', type=str, default='./checkpoint/best_model.pth')
parser.add_argument('--pisr_pth_path', type=str, default='./checkpoint/best_model.pth')
parser.add_argument('--infer_data', type=str, default='real')
parser.add_argument('--scale', type=int, default=2)
parser.add_argument('--output', type=str, default='./infer/real')
parser.add_argument("--cuda", action="store_true", default=True, help="use cuda")

args = parser.parse_args()


def infer():
    # print arguments
    print(args)
    torch.backends.cudnn.benchmark = True

    # set GPU
    cuda = args.cuda
    device = torch.device('cuda' if cuda else 'cpu')
    
    # load model
    print("===> Building models")
    pidnet = PIDNet().to(device)
    pidnet.load_state_dict(torch.load(args.pid_pth_path))
    pisrnet = PISRNet().to(device)
    pisrnet.load_state_dict(torch.load(args.pisr_pth_path))

    with torch.no_grad():
        pidnet.eval()
        pisrnet.eval()

        if args.infer_data == 'real':
            for filename in os.listdir(args.infer_path):
                idx = filename.split('.')[0]
                print(f'Processing image {filename}')
                raw_img = cv2.imread(os.path.join(args.infer_path, filename), -1)

                I0 = cv2.cvtColor(raw_img[1::2, 1::2], cv2.COLOR_BAYER_RG2RGB).astype(np.float32) / 65535.0 * 255
                I45 = cv2.cvtColor(raw_img[::2, 1::2], cv2.COLOR_BAYER_RG2RGB).astype(np.float32) / 65535.0 * 255
                I90 = cv2.cvtColor(raw_img[::2, ::2], cv2.COLOR_BAYER_RG2RGB).astype(np.float32) / 65535.0 * 255
                I135 = cv2.cvtColor(raw_img[1::2, ::2], cv2.COLOR_BAYER_RG2RGB).astype(np.float32) / 65535.0 * 255

                input_img = np.concatenate([I0, I45, I90, I135], axis=-1)
                input_img = np.transpose(input_img, (2, 0, 1))
                input_img = torch.from_numpy(input_img).unsqueeze(0).float().to(device)

                _, _, h, w = input_img.shape

                pad_h = (4 - h % 4) % 4  # Padding needed for height
                pad_w = (4 - w % 4) % 4  # Padding needed for width
                
                original_h, original_w = h*2, w*2

                input_img = F.pad(input_img, (0, pad_w, 0, pad_h), mode='reflect')

                _, demosaic_out = pidnet(input_img)

                demosaic_out = demosaic_out[:, :, :original_h, :original_w]

                _, sr_out = pisrnet(demosaic_out)

                de_I0, de_I45, de_I90, de_I135, de_S0, de_S1, de_S2, de_AoP, de_DoP = get_img_from_tensor(demosaic_out)
                sr_I0, sr_I45, sr_I90, sr_I135, sr_S0, sr_S1, sr_S2, sr_AoP, sr_DoP = get_img_from_tensor(sr_out)

                de_AoP_map, de_DoP_map = visulize_aop_dop(de_AoP, de_DoP)
                sr_AoP_map, sr_DoP_map = visulize_aop_dop(sr_AoP, sr_DoP)

                PID_save_path = os.path.join(args.output, 'PID', idx)
                PISR_save_path = os.path.join(args.output, 'PISR', idx)

                if not os.path.exists(PID_save_path):
                    os.makedirs(PID_save_path)

                if not os.path.exists(PISR_save_path):
                    os.makedirs(PISR_save_path)

                cv2.imwrite(os.path.join(PID_save_path, 'RGB_0.png'), np.uint16(de_I0 * 65535))
                cv2.imwrite(os.path.join(PID_save_path, 'RGB_45.png'), np.uint16(de_I45 * 65535))
                cv2.imwrite(os.path.join(PID_save_path, 'RGB_90.png'), np.uint16(de_I90 * 65535))
                cv2.imwrite(os.path.join(PID_save_path, 'RGB_135.png'), np.uint16(de_I135 * 65535))

                cv2.imwrite(os.path.join(PID_save_path, 'aop.png'), de_AoP_map)
                cv2.imwrite(os.path.join(PID_save_path, 'dop.png'), de_DoP_map)

                cv2.imwrite(os.path.join(PISR_save_path, 'RGB_0.png'), np.uint16(sr_I0 * 65535))
                cv2.imwrite(os.path.join(PISR_save_path, 'RGB_45.png'), np.uint16(sr_I45 * 65535))
                cv2.imwrite(os.path.join(PISR_save_path, 'RGB_90.png'), np.uint16(sr_I90 * 65535))
                cv2.imwrite(os.path.join(PISR_save_path, 'RGB_135.png'), np.uint16(sr_I135 * 65535))

                cv2.imwrite(os.path.join(PISR_save_path, 'aop.png'), sr_AoP_map)
                cv2.imwrite(os.path.join(PISR_save_path, 'dop.png'), sr_DoP_map)

        elif args.infer_data == 'syn':
            PID_PSNR_S0 = 0
            PID_SSIM_S0 = 0
            PID_PSNR_DoP = 0
            PID_SSIM_DoP = 0
            PID_AoP_MAE = 0

            PISR_PSNR_S0 = 0
            PISR_SSIM_S0 = 0
            PISR_PSNR_DoP = 0
            PISR_SSIM_DoP = 0
            PISR_AoP_MAE = 0

            for filename in os.listdir(args.infer_path):
                print(f'Processing image {filename}')
                I0 = cv2.imread(os.path.join(args.infer_path, filename, '0.png'), -1).astype(np.float32) / 65535.0 * 255
                I45 = cv2.imread(os.path.join(args.infer_path, filename, '45.png'), -1).astype(np.float32) / 65535.0 * 255
                I90 = cv2.imread(os.path.join(args.infer_path, filename, '90.png'), -1).astype(np.float32) / 65535.0 * 255
                I135 = cv2.imread(os.path.join(args.infer_path, filename, '135.png'), -1).astype(np.float32) / 65535.0 * 255

                gt_hr = np.concatenate([I0, I45, I90, I135], axis=-1)
                gt_hr = np.transpose(gt_hr, (2, 0, 1))
                gt_hr = torch.from_numpy(gt_hr).unsqueeze(0).float().to(device)

                I0_lr = cv2.resize(I0, (I0.shape[1] // 2, I0.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
                I45_lr = cv2.resize(I45, (I45.shape[1] // 2, I45.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
                I90_lr = cv2.resize(I90, (I90.shape[1] // 2, I90.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
                I135_lr = cv2.resize(I135, (I135.shape[1] // 2, I135.shape[0] // 2), interpolation=cv2.INTER_LINEAR)

                gt_lr = np.concatenate([I0_lr, I45_lr, I90_lr, I135_lr], axis=-1)
                gt_lr = np.transpose(gt_lr, (2, 0, 1))
                gt_lr = torch.from_numpy(gt_lr).unsqueeze(0).float().to(device)

                raw_img = color_polar_mosaic(I0, I45, I90, I135, COLOR_BAYER_PATTERN='RGGB')

                input_0 = cv2.cvtColor(raw_img[1::2, 1::2], cv2.COLOR_BAYER_RG2RGB)
                input_45 = cv2.cvtColor(raw_img[::2, 1::2], cv2.COLOR_BAYER_RG2RGB)
                input_90 = cv2.cvtColor(raw_img[::2, ::2], cv2.COLOR_BAYER_RG2RGB)
                input_135 = cv2.cvtColor(raw_img[1::2, ::2], cv2.COLOR_BAYER_RG2RGB)

                input_img = np.concatenate([input_0, input_45, input_90, input_135], axis=-1)
                input_img = np.transpose(input_img, (2, 0, 1))
                input_img = torch.from_numpy(input_img).unsqueeze(0).float().to(device)

                _, demosaic_out = pidnet(input_img)
                _, sr_out = pisrnet(demosaic_out)

                demosaic_0, demosaic_45, demosaic_90, demosaic_135, demosaic_S0, demosaic_S1, demosaic_S2, demosaic_AoP, demosaic_DoP = get_img_from_tensor(demosaic_out)
                sr_0, sr_45, sr_90, sr_135, sr_S0, sr_S1, sr_S2, sr_AoP, sr_DoP = get_img_from_tensor(sr_out)
                lr_0, lr_45, lr_90, lr_135, lr_S0, lr_S1, lr_S2, lr_aop, lr_dop = get_img_from_tensor(gt_hr)
                hr_0, hr_45, hr_90, hr_135, hr_S0, hr_S1, hr_S2, hr_aop, hr_dop = get_img_from_tensor(gt_hr)

                demosaic_AoP_map, demosaic_DoP_map = visulize_aop_dop(demosaic_AoP, demosaic_DoP)
                sr_AoP_map, sr_DoP_map = visulize_aop_dop(sr_AoP, sr_DoP)

                psnr_s0 = psnr(demosaic_S0, lr_S0, data_range=2)
                PID_PSNR_S0 += psnr_s0
                ssim_s0 = ssim(demosaic_S0, lr_S0, data_range=2, multichannel=True, channel_axis=-1)
                PID_SSIM_S0 += ssim_s0
                psnr_dop = psnr(demosaic_DoP, lr_dop, data_range=1)
                PID_PSNR_DoP += psnr_dop
                ssim_dop = ssim(demosaic_DoP, lr_dop, data_range=1, multichannel=True, channel_axis=-1)
                PID_SSIM_DoP += ssim_dop
                aop_mae = np.mean(np.minimum(np.abs(lr_aop - demosaic_AoP), np.abs(np.abs(lr_aop - demosaic_AoP) - np.pi))) / np.pi * 180
                PID_AoP_MAE += aop_mae

                print('=====> PID Results <=====')
                print(f'PSNR_S0: {psnr_s0:.2f}, SSIM_S0: {ssim_s0:.4f}, PSNR_DoP: {psnr_dop:.2f}, SSIM_DoP: {ssim_dop:.4f}, AoP_MAE: {aop_mae:.4f}')

                psnr_s0 = psnr(sr_S0, hr_S0, data_range=2)
                PISR_PSNR_S0 += psnr_s0
                ssim_s0 = ssim(sr_S0, hr_S0, data_range=2, multichannel=True, channel_axis=-1)
                PISR_SSIM_S0 += ssim_s0
                psnr_dop = psnr(sr_DoP, hr_dop, data_range=1)
                PISR_PSNR_DoP += psnr_dop
                ssim_dop = ssim(sr_DoP, hr_dop, data_range=1, multichannel=True, channel_axis=-1)
                PISR_SSIM_DoP += ssim_dop
                aop_mae = np.mean(np.minimum(np.abs(hr_aop - sr_AoP), np.abs(np.abs(hr_aop - sr_AoP) - np.pi))) / np.pi * 180
                PISR_AoP_MAE += aop_mae

                print('=====> PISR Results <=====')
                print(f'PSNR_S0: {psnr_s0:.2f}, SSIM_S0: {ssim_s0:.4f}, PSNR_DoP: {psnr_dop:.2f}, SSIM_DoP: {ssim_dop:.4f}, AoP_MAE: {aop_mae:.4f}')

                if not os.path.exists(os.path.join(args.output, 'PID', filename)):
                    os.makedirs(os.path.join(args.output, 'PID', filename))

                if not os.path.exists(os.path.join(args.output, 'PISR', filename)):
                    os.makedirs(os.path.join(args.output, 'PISR', filename))

                cv2.imwrite(os.path.join(args.output, 'PID', filename, f'S0.png'), np.uint16(demosaic_S0 / 2 * 65535))
                cv2.imwrite(os.path.join(args.output, 'PID', filename, f'AoP.png'), demosaic_AoP_map)
                cv2.imwrite(os.path.join(args.output, 'PID', filename, f'DoP.png'), demosaic_DoP_map)

                cv2.imwrite(os.path.join(args.output, 'PISR', filename, f'S0.png'), np.uint16(sr_S0 / 2 * 65535))
                cv2.imwrite(os.path.join(args.output, 'PISR', filename, f'AoP.png'), sr_AoP_map)
                cv2.imwrite(os.path.join(args.output, 'PISR', filename, f'DoP.png'), sr_DoP_map)

            PID_PSNR_S0 /= len(os.listdir(args.infer_path))
            PID_SSIM_S0 /= len(os.listdir(args.infer_path))
            PID_PSNR_DoP /= len(os.listdir(args.infer_path))
            PID_SSIM_DoP /= len(os.listdir(args.infer_path))
            PID_AoP_MAE /= len(os.listdir(args.infer_path))

            PISR_PSNR_S0 /= len(os.listdir(args.infer_path))
            PISR_SSIM_S0 /= len(os.listdir(args.infer_path))
            PISR_PSNR_DoP /= len(os.listdir(args.infer_path))
            PISR_SSIM_DoP /= len(os.listdir(args.infer_path))
            PISR_AoP_MAE /= len(os.listdir(args.infer_path))

            print('=====> PID Average Results <=====')
            print(f'PSNR_S0: {PID_PSNR_S0:.2f}, SSIM_S0: {PID_SSIM_S0:.4f}, PSNR_DoP: {PID_PSNR_DoP:.2f}, SSIM_DoP: {PID_SSIM_DoP:.4f}, AoP_MAE: {PID_AoP_MAE:.4f}')

            print('=====> PISR Average Results <=====')
            print(f'PSNR_S0: {PISR_PSNR_S0:.2f}, SSIM_S0: {PISR_SSIM_S0:.4f}, PSNR_DoP: {PISR_PSNR_DoP:.2f}, SSIM_DoP: {PISR_SSIM_DoP:.4f}, AoP_MAE: {PISR_AoP_MAE:.4f}')


if __name__ == '__main__':
    infer()
