import torch
import torch.utils.data as data
import os
import cv2
import numpy as np
from tqdm import tqdm


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

def np2tensor(numpy_array):
    # H, W, C -> C, H, W
    np_transpose = np.ascontiguousarray(numpy_array.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()

    return tensor


class PIDSRData(data.Dataset):
    def __init__(self, args, is_train):
        self.args = args
        self.is_train = is_train
        self.patch_size = self.args.patch_size
        if self.is_train:
            self.root = self.args.train_path
        else:
            self.root = self.args.test_path
        print(f"Dataset path: {self.root}")
        self.input, self.gt_llr, self.gt_lr, self.gt_hr = self.load_file()

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, index):
        input_img = self.input[index]
        gt_llr_img = self.gt_llr[index]
        gt_lr_img = self.gt_lr[index]
        gt_hr_img = self.gt_hr[index]
        # print(input_img.shape, gt_lr_img.shape, gt_hr_img.shape)
        input_tensor = np2tensor(input_img)
        gt_llr_tensor = np2tensor(gt_llr_img)
        gt_lr_tensor = np2tensor(gt_lr_img)
        gt_hr_tensor = np2tensor(gt_hr_img)

        return input_tensor, gt_llr_tensor, gt_lr_tensor, gt_hr_tensor
    
    def data_augment(self, img, patch_size):
        h, w, c = img.shape

        patch_list = []

        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                if i + patch_size <= h and j + patch_size <= w:
                    patch = img[i:i + patch_size, j:j + patch_size, :]
                    patch_list.append(patch)

        flip_list = []
        for patch in patch_list:
            flip_list.append(patch)
            # flip left-right, up-down, rotate 90 degree
            flip_list.append(np.flip(patch, axis=1))
            flip_list.append(np.flip(patch, axis=0))
            flip_list.append(np.rot90(patch, k=-1, axes=(0, 1)))

        return flip_list
    
    def polar_mosaic(self, rgb_0, rgb_45, rgb_90, rgb_135, COLOR_BAYER_PATTERN='RGGB'):
        # add bayer pattern
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

        bayer_0 = mosaic_img[1::2, 1::2]
        bayer_45 = mosaic_img[::2, 1::2]
        bayer_90 = mosaic_img[::2, ::2]
        bayer_135 = mosaic_img[1::2, ::2]

        return bayer_0, bayer_45, bayer_90, bayer_135
        
    def load_file(self):
        input_list = []
        gt_llr_list = []
        gt_lr_list = []
        gt_hr_list = []

        if self.is_train:
            for filename in tqdm(os.listdir(self.root)):
                img_path = os.path.join(self.root, filename)
                # read 4 images
                I0 = cv2.imread(os.path.join(img_path, 'RGB_0.png'), -1).astype(np.float32) / 65535.0
                I45 = cv2.imread(os.path.join(img_path, 'RGB_45.png'), -1).astype(np.float32) / 65535.0
                I90 = cv2.imread(os.path.join(img_path, 'RGB_90.png'), -1).astype(np.float32) / 65535.0
                I135 = cv2.imread(os.path.join(img_path, 'RGB_135.png'), -1).astype(np.float32) / 65535.0

                img_pack = np.concatenate([I0, I45, I90, I135], axis=-1)
                # data augmentation
                img_patches = self.data_augment(img_pack, self.patch_size)

                for patch in img_patches:
                    gt_hr_0 = patch[..., 0:3] * self.args.rgb_range
                    gt_hr_45 = patch[..., 3:6] * self.args.rgb_range
                    gt_hr_90 = patch[..., 6:9] * self.args.rgb_range
                    gt_hr_135 = patch[..., 9:12] * self.args.rgb_range
                    gt_hr_list.append(np.concatenate([gt_hr_0, gt_hr_45, gt_hr_90, gt_hr_135], axis=-1))

                    gt_lr_0 = cv2.resize(gt_hr_0, (gt_hr_0.shape[1] // 2, gt_hr_0.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
                    gt_lr_45 = cv2.resize(gt_hr_45, (gt_hr_45.shape[1] // 2, gt_hr_45.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
                    gt_lr_90 = cv2.resize(gt_hr_90, (gt_hr_90.shape[1] // 2, gt_hr_90.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
                    gt_lr_135 = cv2.resize(gt_hr_135, (gt_hr_135.shape[1] // 2, gt_hr_135.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
                    gt_lr_list.append(np.concatenate([gt_lr_0, gt_lr_45, gt_lr_90, gt_lr_135], axis=-1))

                    gt_llr_0 = cv2.resize(gt_hr_0, (gt_hr_0.shape[1] // 4, gt_hr_0.shape[0] // 4), interpolation=cv2.INTER_LINEAR)
                    gt_llr_45 = cv2.resize(gt_hr_45, (gt_hr_45.shape[1] // 4, gt_hr_45.shape[0] // 4), interpolation=cv2.INTER_LINEAR)
                    gt_llr_90 = cv2.resize(gt_hr_90, (gt_hr_90.shape[1] // 4, gt_hr_90.shape[0] // 4), interpolation=cv2.INTER_LINEAR)
                    gt_llr_135 = cv2.resize(gt_hr_135, (gt_hr_135.shape[1] // 4, gt_hr_135.shape[0] // 4), interpolation=cv2.INTER_LINEAR)
                    gt_llr_list.append(np.concatenate([gt_llr_0, gt_llr_45, gt_llr_90, gt_llr_135], axis=-1))

                    bayer_0, bayer_45, bayer_90, bayer_135 = self.polar_mosaic(gt_lr_0, gt_lr_45, gt_lr_90, gt_lr_135, 'RGGB')

                    input_0 = cv2.cvtColor(bayer_0, cv2.COLOR_BAYER_RG2RGB)
                    input_45 = cv2.cvtColor(bayer_45, cv2.COLOR_BAYER_RG2RGB)
                    input_90 = cv2.cvtColor(bayer_90, cv2.COLOR_BAYER_RG2RGB)
                    input_135 = cv2.cvtColor(bayer_135, cv2.COLOR_BAYER_RG2RGB)
                    input_list.append(np.concatenate([input_0, input_45, input_90, input_135], axis=-1))

        else:
            # 读取测试数据
            for filename in tqdm(os.listdir(self.root)):
                img_path = os.path.join(self.root, filename)
                # crop region for test
                rgb_0 = cv2.imread(os.path.join(img_path, 'RGB_0.png'), -1)[500:1524:, 700:1724, :].astype(np.float32) / 65535.0
                rgb_45 = cv2.imread(os.path.join(img_path, 'RGB_45.png'), -1)[500:1524:, 700:1724, :].astype(np.float32) / 65535.0
                rgb_90 = cv2.imread(os.path.join(img_path, 'RGB_90.png'), -1)[500:1524:, 700:1724, :].astype(np.float32) / 65535.0
                rgb_135 = cv2.imread(os.path.join(img_path, 'RGB_135.png'), -1)[500:1524:, 700:1724, :].astype(np.float32) / 65535.0

                I0 = rgb_0 * self.args.rgb_range
                I45 = rgb_45 * self.args.rgb_range
                I90 = rgb_90 * self.args.rgb_range
                I135 = rgb_135 * self.args.rgb_range

                gt_hr_list.append(np.concatenate([I0, I45, I90, I135], axis=-1))

                I0_lr = cv2.resize(I0, (I0.shape[1] // 2, I0.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
                I45_lr = cv2.resize(I45, (I45.shape[1] // 2, I45.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
                I90_lr = cv2.resize(I90, (I90.shape[1] // 2, I90.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
                I135_lr = cv2.resize(I135, (I135.shape[1] // 2, I135.shape[0] // 2), interpolation=cv2.INTER_LINEAR)

                gt_lr_list.append(np.concatenate([I0_lr, I45_lr, I90_lr, I135_lr], axis=-1))

                I0_llr = cv2.resize(I0, (I0.shape[1] // 4, I0.shape[0] // 4), interpolation=cv2.INTER_LINEAR)
                I45_llr = cv2.resize(I45, (I45.shape[1] // 4, I45.shape[0] // 4), interpolation=cv2.INTER_LINEAR)
                I90_llr = cv2.resize(I90, (I90.shape[1] // 4, I90.shape[0] // 4), interpolation=cv2.INTER_LINEAR)
                I135_llr = cv2.resize(I135, (I135.shape[1] // 4, I135.shape[0] // 4), interpolation=cv2.INTER_LINEAR)

                gt_llr_list.append(np.concatenate([I0_llr, I45_llr, I90_llr, I135_llr], axis=-1))

                bayer_0, bayer_45, bayer_90, bayer_135 = self.polar_mosaic(I0_lr, I45_lr, I90_lr, I135_lr, 'RGGB')

                input_0 = cv2.cvtColor(bayer_0, cv2.COLOR_BAYER_RG2RGB)
                input_45 = cv2.cvtColor(bayer_45, cv2.COLOR_BAYER_RG2RGB)
                input_90 = cv2.cvtColor(bayer_90, cv2.COLOR_BAYER_RG2RGB)
                input_135 = cv2.cvtColor(bayer_135, cv2.COLOR_BAYER_RG2RGB)

                input_list.append(np.concatenate([input_0, input_45, input_90, input_135], axis=-1))

        return input_list, gt_llr_list, gt_lr_list, gt_hr_list
