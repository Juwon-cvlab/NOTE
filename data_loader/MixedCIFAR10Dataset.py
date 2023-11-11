import os
import warnings
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import pandas as pd
import time
import numpy as np
import sys
import conf

opt = conf.CIFAR10Opt


class MixedCIFAR10Dataset(torch.utils.data.Dataset):

    def __init__(self, file='../dataset/ichar/minmax_scaling_all.csv',
                 domains=None, activities=None,
                 max_source=100, transform='none'):
        st = time.time()
        self.domains = domains
        self.activity = activities
        self.max_source = max_source

        self.domains = domains

        self.img_shape = opt['img_size']
        self.features = None
        self.class_labels = None
        self.domain_labels = None
        self.file_path = opt['file_path']

        self.label_filename = 'labels.npy'

        self.sub_path_list1 = []
        self.sub_path_list2 = []
        self.data_filename_list = []

        self.num_domains = len(domains)

        assert (len(domains) > 0)
        for domain in domains:
            if domain.startswith('original'):
                self.sub_path_list1.append('origin')
                self.sub_path_list2.append('')
                self.data_filename_list.append('original.npy')
            elif domain.startswith('test'):
                self.sub_path_list1.append('corrupted')
                self.sub_path_list2.append('severity-1')  # all data are same in 1~5
                self.data_filename_list.append('test.npy')
            elif domain.endswith('-1'):
                self.sub_path_list1.append('corrupted')
                self.sub_path_list2.append('severity-1')
                self.data_filename_list.append(domain.split('-')[0] + '.npy')
            elif domain.endswith('-2'):
                self.sub_path_list1.append('corrupted')
                self.sub_path_list2.append('severity-2')
                self.data_filename_list.append(domain.split('-')[0] + '.npy')
            elif domain.endswith('-3'):
                self.sub_path_list1.append('corrupted')
                self.sub_path_list2.append('severity-3')
                self.data_filename_list.append(domain.split('-')[0] + '.npy')
            elif domain.endswith('-4'):
                self.sub_path_list1.append('corrupted')
                self.sub_path_list2.append('severity-4')
                self.data_filename_list.append(domain.split('-')[0] + '.npy')
            elif domain.endswith('-5'):
                self.sub_path_list1.append('corrupted')
                self.sub_path_list2.append('severity-5')
                self.data_filename_list.append(domain.split('-')[0] + '.npy')
            elif domain.endswith('-all'):
                self.sub_path_list1.append('corrupted')
                self.sub_path_list2.append('severity-all')
                self.data_filename_list.append(domain.split('-')[0] + '.npy')
            else:
                raise NotImplemented

        if transform == 'src':
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ])

        elif transform == 'val':
            self.transform = None

        # From EcoTTA paper
        elif transform == 'aug-v1':
            self.transform = transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.4),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 2.0))], p=0.2),
                transforms.RandomGrayscale(p=0.1)
            ])

        # From SWR paper
        elif transform == 'aug-v2':
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
                transforms.RandomChoice(transforms=[transforms.RandomGrayscale(p=0.5), transforms.RandomInvert(p=0.5)], p=[0.5, 0.5]),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3,3), sigma=(1.0, 2.0))], p=0.5)
            ])

        elif transform == 'aug-v3':
            self.transform = transforms.Compose([
                transforms.RandomGrayscale(p=0.8),
                transforms.RandomInvert(p=0.8)
            ])
        
        elif transform == 'aug-v4':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
            ])

        else:
            raise NotImplementedError

        self.preprocessing()

    def preprocessing(self):

        features = []

        for idx, (sub_path1, sub_path2, data_filename) in enumerate(zip(self.sub_path_list1, self.sub_path_list2, self.data_filename_list)):

            path = f'{self.file_path}/{sub_path1}/{sub_path2}/'
            print(path, data_filename)
            data = np.load(path + data_filename)
            data = np.transpose(data, (0, 3, 1, 2))
            data = data.astype(np.float32) / 255.0

            features.append(data)
            
            if idx == 0:
                self.class_labels = np.load(path + self.label_filename)
            
        num_samples = len(self.class_labels)

        result_feats = []
        result_do_labels = []
        
        for data_idx in range(num_samples):
            domain_idx = np.random.randint(self.num_domains)
            result_feats.append(features[domain_idx][data_idx])
            result_do_labels.append(domain_idx)

        self.features = np.array(result_feats)
        self.domain_labels = np.array(result_do_labels)

        self.dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(self.features),
            torch.from_numpy(self.class_labels),
            torch.from_numpy(self.domain_labels))

    def __len__(self):
        return len(self.dataset)

    def get_num_domains(self):
        return len(self.domains)

    def get_datasets_per_domain(self):
        return self.datasets

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        img, cl, dl = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, cl, dl


if __name__ == '__main__':
    pass
