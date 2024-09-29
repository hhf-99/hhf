from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import os
class EFID(torch.utils.data.Dataset):
    def __init__(self, ref_path, eha_path, txt_file_name, transform=None, resize=False, size=None, flip=False):
        super(EFID, self).__init__()
        self.ref_path = ref_path
        self.eha_path = eha_path
        self.txt_file_name = txt_file_name
        self.transform = transform
        self.flip = flip
        self.resize = resize
        self.size = size
        ref_files_data, eha_files_data, score_data = [], [], []

        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                eha, score = line[:-1].split(',')
                ref = eha[7:13] + '.png'
                score = float(score)
                ref_files_data.append(ref)
                eha_files_data.append(eha)
                score_data.append(score)

        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = score_data.astype('float').reshape(-1, 1)

        self.data_dict = {
            'r_img_list': ref_files_data,
            'd_img_list': eha_files_data,
            'score_list': score_data
        }

    def normalization(self, data):
        range_value = np.max(data) - np.min(data)
        return (data - np.min(data)) / range_value

    def __len__(self):
        return len(self.data_dict['r_img_list'])

    def __getitem__(self, idx):
        r_img_name = self.data_dict['r_img_list'][idx]
        r_img = Image.open(os.path.join(self.ref_path, r_img_name)).convert('RGB')

        if self.flip:
            r_img = transforms.functional.hflip(r_img)
        if self.resize:
            r_img = r_img.resize(self.size)

        r_img = transforms.ToTensor()(r_img)
        r_img = (r_img - 0.5) / 0.5

        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = Image.open(os.path.join(self.eha_path, d_img_name)).convert('RGB')

        if self.flip:
            d_img = transforms.functional.hflip(d_img)
        if self.resize:
            d_img = d_img.resize(self.size)

        d_img = transforms.ToTensor()(d_img)
        d_img = (d_img - 0.5) / 0.5

        score = self.data_dict['score_list'][idx]
        sample = {
            'r_img_org': r_img,
            'd_img_org': d_img,
            'score': score,
            'd_img_name': d_img_name
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
