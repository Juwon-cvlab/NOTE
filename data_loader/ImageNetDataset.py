import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import conf

opt = conf.ImageNetOpt

corruption_dict = {'gaussian_noise': 'noise',   'shot_noise': 'noise',          'impulse_noise': 'noise', 
                   'defocus_blur': 'blur',      'glass_blur' : 'blur',          'motion_blur': 'blur',      'zoom_blur': 'blur', 
                   'snow': 'weather',           'frost':'weather',              'fog':'weather',            'brightness' :'weather',
                   'contrast':'digital',        'elastic_transform':'digital',  'pixelate':'digital',       'jpeg_compression':'digital'}


class ImageNetDataset(datasets.ImageFolder):

    def __init__(self, domains=None, transform='none'):
        
        assert (len(domains) > 0)
        if domains[0].startswith('original'):
            raise NotImplemented
        elif domains[0].startswith('test'):
            raise NotImplemented
        elif domains[0].endswith('-all'):
            raise NotImplemented
        else:
            corruption_type = domains[0].split('-')[0]
            level = domains[0].split('-')[1]
            super_type = corruption_dict[corruption_type]

        root = opt['file_path']
        path = os.path.join(root, super_type, corruption_type, level)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if transform == 'train':
            transform = transforms.Compose([transforms.RandomResizedCrop(224),
									transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
									transforms.ToTensor(),
									normalize
                                    ])
        elif transform == 'test_v1':
             transform = transforms.Compose([transforms.Resize(256),
									transforms.CenterCrop(224),
									transforms.ToTensor(),
									normalize
                                    ])
        elif transform == 'test_v2':
            transform = transforms.Compose([transforms.CenterCrop(224),
									transforms.ToTensor(),
									normalize
                                    ])
        else:
            raise NotImplemented
        
        super(ImageNetDataset, self).__init__(path, transform)

        self.domains = domains
        self.file_path = root

    def get_num_domains(self):
        return len(self.domains)

    # def get_datasets_per_domain(self):
    #     return self.datasets

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img_input = self.loader(path)

        if self.transform is not None:
            img = self.transform(img_input)
        else:
            img = img_input
        
        return img, target, 0


if __name__ == '__main__':
    pass
