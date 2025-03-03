import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def visulize_aop_dop(aop, dop):
    aop = (aop + np.pi / 2) / np.pi * 255
    aop_map = cv2.applyColorMap(cv2.cvtColor(aop.astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)

    dop_map = cv2.applyColorMap(cv2.cvtColor((dop * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)

    return aop_map, dop_map

def calculate_stokes(img_tensor, max_val=1):
    img_numpy = np.transpose(img_tensor.squeeze(0).cpu().numpy(), (1, 2, 0))
    if max_val == 1:
        img_0 = np.clip(img_numpy[:, :, 0:3], 0, 1)
        img_45 = np.clip(img_numpy[:, :, 3:6], 0, 1)
        img_90 = np.clip(img_numpy[:, :, 6:9], 0, 1)
        img_135 = np.clip(img_numpy[:, :, 9:12], 0, 1)
    else:
        img_0 = np.clip(img_numpy[:, :, 0:3], 0, 255) / 255
        img_45 = np.clip(img_numpy[:, :, 3:6], 0, 255) / 255
        img_90 = np.clip(img_numpy[:, :, 6:9], 0, 255) / 255
        img_135 = np.clip(img_numpy[:, :, 9:12], 0, 255) / 255

    S0 = (img_0 + img_45 + img_90 + img_135) / 2
    S1 = img_0 - img_90
    S2 = img_45 - img_135

    aop = np.arctan2(S2, S1) / 2
    dop = np.clip(np.sqrt(S1 **2 + S2 **2) / (S0 + 1e-7), 0, 1)

    return img_0, img_45, img_90, img_135, S0, S1, S2, aop, dop

def compute_image_gradients(img):
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    sobel_x = sobel_x.to(img.device).repeat(img.shape[1], 1, 1, 1)
    sobel_y = sobel_y.to(img.device).repeat(img.shape[1], 1, 1, 1)
    
    grad_x = F.conv2d(img, sobel_x, padding=1, groups=img.shape[1])
    grad_y = F.conv2d(img, sobel_y, padding=1, groups=img.shape[1])
    
    return grad_x, grad_y

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.alpha = 0.8
        self.beta = 0.2
    
    def forward(self, generated_img, target_img):
        gen_grad_x, gen_grad_y = compute_image_gradients(generated_img)
        tgt_grad_x, tgt_grad_y = compute_image_gradients(target_img)
        
        loss_x = self.alpha * F.l1_loss(gen_grad_x, tgt_grad_x) + self.beta * F.l1_loss(generated_img, target_img)
        loss_y = self.alpha * F.l1_loss(gen_grad_y, tgt_grad_y) + self.beta * F.l1_loss(generated_img, target_img)
        
        return loss_x + loss_y

def polar_loss(output, gt, device):
    l1loss = nn.L1Loss().to(device)
    gradientloss = GradientLoss().to(device)

    output_0 = output[:, 0:3, :, :]
    output_45 = output[:, 3:6, :, :]
    output_90 = output[:, 6:9, :, :]
    output_135 = output[:, 9:12, :, :]

    gt_0 = gt[:, 0:3, :, :]
    gt_45 = gt[:, 3:6, :, :]
    gt_90 = gt[:, 6:9, :, :]
    gt_135 = gt[:, 9:12, :, :]

    output_S0 = (output_0 + output_45 + output_90 + output_135) / 2
    output_S1 = output_0 - output_90
    output_S2 = output_45 - output_135

    gt_S0 = (gt_0 + gt_45 + gt_90 + gt_135) / 2
    gt_S1 = gt_0 - gt_90
    gt_S2 = gt_45 - gt_135

    epsilon = 1e-6

    output_aop = torch.atan2(output_S2 + epsilon, output_S1 + epsilon) / 2
    output_dop = torch.clamp(torch.sqrt(output_S1 **2 + output_S2 **2 + epsilon) / (output_S0 + epsilon), 0, 1)

    gt_aop = torch.atan2(gt_S2 + epsilon, gt_S1 + epsilon) / 2
    gt_dop = torch.clamp(torch.sqrt(gt_S1 **2 + gt_S2 **2 + epsilon) / (gt_S0 + epsilon), 0, 1)

    img_loss = gradientloss(output, gt) + gradientloss(output_S0, gt_S0)
    stokes_loss = l1loss(output_S1, gt_S1) + l1loss(output_S2, gt_S2)
    aop_loss = l1loss(output_aop, gt_aop) + l1loss(output_dop, gt_dop)
    polar_loss = l1loss(output_0 + output_90, output_45 + output_135)

    return img_loss + stokes_loss * 10 + aop_loss * 10 + polar_loss * 10

def get_img_from_tensor(output_tensor):
    output_0 = np.transpose(output_tensor[:, 0:3 :, :].squeeze(0).cpu().numpy(), (1, 2, 0))
    output_45 = np.transpose(output_tensor[:, 3:6 :, :].squeeze(0).cpu().numpy(), (1, 2, 0))
    output_90 = np.transpose(output_tensor[:, 6:9 :, :].squeeze(0).cpu().numpy(), (1, 2, 0))
    output_135 = np.transpose(output_tensor[:, 9:12 :, :].squeeze(0).cpu().numpy(), (1, 2, 0))

    I0 = np.clip(output_0, 0, 255)
    I45 = np.clip(output_45, 0, 255)
    I90 = np.clip(output_90, 0, 255)
    I135 = np.clip(output_135, 0, 255)

    I0 = np.float32(np.uint16(I0 / 255 * 65535)) / 65535
    I45 = np.float32(np.uint16(I45 / 255 * 65535)) / 65535
    I90 = np.float32(np.uint16(I90 / 255 * 65535)) / 65535
    I135 = np.float32(np.uint16(I135 / 255 * 65535)) / 65535

    S0 = (I0 + I45 + I90 + I135) / 2
    S1 = I0 - I90
    S2 = I45 - I135

    AoP = np.arctan2(S2, S1) / 2
    DoP = np.clip(np.sqrt(S1 ** 2 + S2 ** 2) / (S0 + 1e-8), 0, 1)

    return I0, I45, I90, I135, S0, S1, S2, AoP, DoP


def color_polar_mosaic(rgb_0, rgb_45, rgb_90, rgb_135, COLOR_BAYER_PATTERN='RGGB'):
    mosaic_img = np.zeros(shape=(rgb_0.shape[0], rgb_0.shape[1]), dtype=np.uint8)

    if COLOR_BAYER_PATTERN == 'BGGR':
        color = [0, 1, 1, 2]
    elif COLOR_BAYER_PATTERN == 'RGGB':
        color = [2, 1, 1, 0]
    elif COLOR_BAYER_PATTERN == 'GBRG':
        color = [1, 0, 2, 1]
    elif COLOR_BAYER_PATTERN == 'GRBG':
        color = [1, 2, 0, 1]
    else:
        raise ValueError('Invalid COLOR_BAYER_PATTERN')

    mosaic_img[::4, ::4] = rgb_90[::4, ::4, color[0]]
    mosaic_img[::4, 1::4] = rgb_45[::4, 1::4, color[0]]
    mosaic_img[1::4, ::4] = rgb_135[1::4, ::4, color[0]]
    mosaic_img[1::4, 1::4] = rgb_0[1::4, 1::4, color[0]]

    mosaic_img[::4, 2::4] = rgb_90[::4, 2::4, color[1]]
    mosaic_img[::4, 3::4] = rgb_45[::4, 3::4, color[1]]
    mosaic_img[1::4, 2::4] = rgb_135[1::4, 2::4, color[1]]
    mosaic_img[1::4, 3::4] = rgb_0[1::4, 3::4, color[1]]

    mosaic_img[2::4, ::4] = rgb_90[2::4, ::4, color[2]]
    mosaic_img[2::4, 1::4] = rgb_45[2::4, 1::4, color[2]]
    mosaic_img[3::4, ::4] = rgb_135[3::4, ::4, color[2]]
    mosaic_img[3::4, 1::4] = rgb_0[3::4, 1::4, color[2]]

    mosaic_img[2::4, 2::4] = rgb_90[2::4, 2::4, color[3]]
    mosaic_img[2::4, 3::4] = rgb_45[2::4, 3::4, color[3]]
    mosaic_img[3::4, 2::4] = rgb_135[3::4, 2::4, color[3]]
    mosaic_img[3::4, 3::4] = rgb_0[3::4, 3::4, color[3]]

    return mosaic_img

def pad_image(image):
    h, w = image.shape[:2]
    pad_h = 8 - (h % 8) if h % 8 != 0 else 0
    pad_w = 8 - (w % 8) if w % 8 != 0 else 0

    if pad_h > 0 or pad_w > 0:
        top = bottom = pad_h // 2
        left = right = pad_w // 2
        if pad_h % 2 != 0:  
            bottom += 1
        if pad_w % 2 != 0: 
            right += 1

        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return image
