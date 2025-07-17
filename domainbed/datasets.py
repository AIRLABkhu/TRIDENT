# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import glob
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset, ConcatDataset, Dataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate

from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "PACS_dom",
    "target_augmented_PACS",
    "augmented_PACS",
    "aug_PACS",
    "aug_VLCS",
    "aug_OfficeHome",
    "augmented_VLCS",
    "augmented_OfficeHome",
    "CrossDomainDataset",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",
    # Spawrious datasets
    "SpawriousO2O_easy",
    "SpawriousO2O_medium",
    "SpawriousO2O_hard",
    "SpawriousM2M_easy",
    "SpawriousM2M_medium",
    "SpawriousM2M_hard",
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    if "aug_" in dataset_name:
        return 4 
    else:
        return len(get_dataset_class(dataset_name).ENVIRONMENTS)

def classwise_undersample(dataset: ImageFolder, ratio: float, seed: int = 42,
                          min_samples_per_class: int = 1, verbose: bool = True):
    from collections import defaultdict
    from torch.utils.data import Subset
    ratio = float(ratio)
    class_to_indices = defaultdict(list)
    for idx, (_, class_idx) in enumerate(dataset.samples):
        class_to_indices[class_idx].append(idx)

    sampled_indices = []
    generator = torch.Generator().manual_seed(seed)

    for class_idx, indices in class_to_indices.items():
        n = len(indices)
        k = max(int(n * ratio), min_samples_per_class)
        k = min(k, n)
        if verbose:
            print(f"Class {class_idx}: total={n}, sampled={k}")
        sampled = torch.tensor(indices)[torch.randperm(n, generator=generator)[:k]]
        sampled_indices.extend(sampled.tolist())

    if verbose:
        print(f"Total sampled: {len(sampled_indices)} images")

    return Subset(dataset, sampled_indices)

def match_classwise_distribution(
        source_dataset: ImageFolder,
        reference_path: str,
        seed: int = 42,
        verbose: bool = True
    ):
    """
    Args:
        source_dataset (ImageFolder): 대상 데이터셋 (샘플링 대상)
        reference_path (str): 클래스 분포를 참조할 CLEANED 폴더 경로
        seed (int): 랜덤 시드
        verbose (bool): 로그 출력 여부
    Returns:
        Subset: 클래스 분포가 참조 폴더와 맞춰진 Subset
    """
    from collections import defaultdict
    from torch.utils.data import Subset
    # 1. 참조 데이터셋 로드
    reference_dataset = ImageFolder(reference_path)

    # 2. 클래스별 개수 추출
    ref_class_counts = defaultdict(int)
    for _, label in reference_dataset.samples:
        ref_class_counts[label] += 1

    if verbose:
        print("Reference class distribution (CLEANED):")
        for c, n in ref_class_counts.items():
            print(f"  Class {c}: {n} samples")

    # 3. 대상 데이터셋 클래스 인덱스 구성
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(source_dataset.samples):
        class_to_indices[label].append(idx)

    # 4. 샘플링 수행
    sampled_indices = []
    generator = torch.Generator().manual_seed(seed)

    for label, indices in class_to_indices.items():
        n_available = len(indices)
        n_target = ref_class_counts.get(label, 0)

        if n_target == 0:
            continue  # 해당 클래스 없음 → 스킵

        if n_available < n_target:
            if verbose:
                print(f"⚠️ Class {label}: only {n_available} available, using all")
            n_target = n_available

        randperm = torch.randperm(n_available, generator=generator)
        selected = torch.tensor(indices)[randperm[:n_target]]
        sampled_indices.extend(selected.tolist())

    if verbose:
        print(f"\n✅ Total selected: {len(sampled_indices)} samples")

    return Subset(source_dataset, sampled_indices)

class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                         self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)

class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):
            path = os.path.join(root, environment)

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = ImageFolder(path,
                transform=env_transform)
            self.num_classes = len(env_dataset.classes)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)

class MultipleEnvironmentAugImageFolder(MultipleDomainDataset):
    def __init__(self, origin, aug, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(origin) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):
            origin_path = os.path.join(origin, environment)
            aug_path = os.path.join(aug, environment)

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform
            
            if i not in test_envs:
                origin_dataset = ImageFolder(origin_path, transform=env_transform)
                aug_dataset = ImageFolder(aug_path, transform=env_transform)
                env_dataset = ConcatDataset([origin_dataset, aug_dataset])
            else:
                env_dataset = ImageFolder(origin_path,
                    transform=env_transform)
                self.num_classes = len(env_dataset.classes)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class CrossDomainDataset(Dataset):
    def __init__(self, root_dir, test_envs, hparams):
        self.root_dir = root_dir
        self.domain1, self.domain2  = test_envs
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.classes = classes
        
        self.domain1_images = self.load_domain_images(domain1)
        self.domain2_images = self.load_domain_images(domain2)
        
        self.balance_classes()
        
    def load_domain_images(self, domain):
        domain_images = {cls: [] for cls in self.classes}
        domain_path = os.path.join(self.root_dir, domain)
        
        for cls in self.classes:
            class_path = os.path.join(domain_path, cls)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                domain_images[cls].append(img_path)
        
        return domain_images
    
    def balance_classes(self):
        for cls in self.classes:
            min_count = min(len(self.domain1_images[cls]), len(self.domain2_images[cls]))
            self.domain1_images[cls] = random.sample(self.domain1_images[cls], min_count)
            self.domain2_images[cls] = random.sample(self.domain2_images[cls], min_count)
    
    def __len__(self):
        return sum(len(self.domain1_images[cls]) for cls in self.classes)
    
    def __getitem__(self, idx):
        cls = random.choice(self.classes)
        domain1_img_path = random.choice(self.domain1_images[cls])
        domain2_img_path = random.choice(self.domain2_images[cls])
        
        domain1_img = Image.open(domain1_img_path).convert('RGB')
        domain2_img = Image.open(domain2_img_path).convert('RGB')
        
        if self.transform:
            domain1_img = self.transform(domain1_img)
            domain2_img = self.transform(domain2_img)
        
        return (domain1_img, domain2_img), self.classes.index(cls)


# class noisy_PACS(MultipleDomainDataset):
    
class augmented_PACS(MultipleDomainDataset):
    CHECKPOINT_FREQ = 100
    ENVIRONMENTS = ["A", "C", "P", "S"]
    # dirs = ['art_painting', 'cartoon', 'photo', 'sketch']
    def __init__(self, root, test_envs,  hparams):
        # super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)
        # def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        ENVIRONMENTS = ["A", "C", "P", "S"]
        augment = hparams['data_augmentation']
        source_data = [component for i, component in enumerate(ENVIRONMENTS) if i not in test_envs]
        source_data = ''.join(source_data)
        train_root = os.path.join(root, hparams['data_augmentation_root'], source_data)
        
        # train_root2 = os.path.join(root, 'PACS_filter')
        test_root = os.path.join(root, 'PACS')
        dirs = os.listdir(test_root)
        dirs = sorted(dirs)
        test_dirs = [dirs[x] for x in test_envs]
        train_dirs = [dirs[x] for x in range(4) if x not in test_envs]
        train_envs = [x for x in range(4) if x not in test_envs]
        # environments = [f.name for f in os.scandir(root) if f.is_dir()]
        # environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(dirs):
            subset = []
            if i in train_envs:
                env_transform = augment_transform if augment else transform
                path = os.path.join(test_root, environment)
                print(f'Train data path: {path}') 
                sub_dataset = ImageFolder(path, transform=env_transform)
                subset.append(sub_dataset)
                # content_dir = os.path.join(train_root, f'PACS_{environment}')
                aug_dir = os.listdir(train_root)
                for aug in aug_dir:
                    if f"2{environment}" in aug:
                        path = os.path.join(train_root, aug)
                        sub_dataset = ImageFolder(path, transform= env_transform)
                        if float(hparams['undersample_ratio']) < 1.0:
                            ratio = hparams['undersample_ratio']
                            sub_dataset = classwise_undersample(sub_dataset, ratio, min_samples_per_class=1, verbose=True)
                        elif hparams['data_randomsample_root'] is not None:
                            sub_dataset = match_classwise_distribution(
                                sub_dataset, os.path.join(root, hparams['data_randomsample_root'], source_data, aug), verbose=True)
                            
                        subset.append(sub_dataset)
                        print(f'Aug {environment} data path: {path}') 

                combined_dataset = ConcatDataset(subset)
                self.datasets.append(combined_dataset)
                 
            else:
                env_transform = transform
                path = os.path.join(test_root, environment)
                sub_dataset = ImageFolder(path, transform=env_transform)
                self.datasets.append(sub_dataset)
                # subset.append(sub_dataset)
                print(f'Test data path: {path}')    

        if isinstance(sub_dataset, Subset):
            self.num_classes = len(sub_dataset.dataset.classes)
        else:
            self.num_classes = len(sub_dataset.classes)

        self.input_shape = (3, 224, 224,)


class augmented_VLCS(MultipleDomainDataset):
    CHECKPOINT_FREQ = 100
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs,  hparams):
        super().__init__()
        ENVIRONMENTS = ["C", "L", "S", "V"]
        augment = hparams['data_augmentation']
        source_data = [component for i, component in enumerate(ENVIRONMENTS) if i not in test_envs]
        source_data = ''.join(source_data)
        train_root = os.path.join(root, hparams['data_augmentation_root'], source_data)
        
        # train_root2 = os.path.join(root, 'PACS_filter')
        test_root = os.path.join(root, 'VLCS')
        dirs = os.listdir(test_root)
        dirs = sorted(dirs)
        test_dirs = [dirs[x] for x in test_envs]
        train_dirs = [dirs[x] for x in range(4) if x not in test_envs]
        train_envs = [x for x in range(4) if x not in test_envs]
        # environments = [f.name for f in os.scandir(root) if f.is_dir()]
        # environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(dirs):
            subset = []
            if i in train_envs:
                env_transform = augment_transform if augment else transform
                path = os.path.join(test_root, environment)
                print(f'Train data path: {path}') 
                sub_dataset = ImageFolder(path, transform=env_transform)
                subset.append(sub_dataset)
                # content_dir = os.path.join(train_root, f'PACS_{environment}')
                aug_dir = os.listdir(train_root)
                for aug in aug_dir:
                    if f"2{environment}" in aug:
                        path = os.path.join(train_root, aug)
                        sub_dataset = ImageFolder(path, transform= env_transform)
                        subset.append(sub_dataset)
                        print(f'Aug {environment} data path: {path}') 

                combined_dataset = ConcatDataset(subset)
                self.datasets.append(combined_dataset)
                 
            else:
                env_transform = transform
                path = os.path.join(test_root, environment)
                sub_dataset = ImageFolder(path, transform=env_transform)
                self.datasets.append(sub_dataset)
                # subset.append(sub_dataset)
                print(f'Test data path: {path}')    

        self.num_classes = len(sub_dataset.classes)

        self.input_shape = (3, 224, 224,)


class augmented_OfficeHome(MultipleDomainDataset):
    CHECKPOINT_FREQ = 100
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs,  hparams):
        super().__init__()
        ENVIRONMENTS = ["A", "C", "P", "R"]
        augment = hparams['data_augmentation']
        source_data = [component for i, component in enumerate(ENVIRONMENTS) if i not in test_envs]
        source_data = ''.join(source_data)
        train_root = os.path.join(root, hparams['data_augmentation_root'], source_data)
        
        # train_root2 = os.path.join(root, 'PACS_filter')
        test_root = os.path.join(root, 'OfficeHome')
        dirs = os.listdir(test_root)
        dirs = sorted(dirs)
        test_dirs = [dirs[x] for x in test_envs]
        train_dirs = [dirs[x] for x in range(4) if x not in test_envs]
        train_envs = [x for x in range(4) if x not in test_envs]
        # environments = [f.name for f in os.scandir(root) if f.is_dir()]
        # environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(dirs):
            subset = []
            if i in train_envs:
                env_transform = augment_transform if augment else transform
                path = os.path.join(test_root, environment)
                print(f'Train data path: {path}') 
                sub_dataset = ImageFolder(path, transform=env_transform)
                subset.append(sub_dataset)
                # content_dir = os.path.join(train_root, f'PACS_{environment}')
                aug_dir = os.listdir(train_root)
                for aug in aug_dir:
                    if f"2{environment}" in aug:
                        path = os.path.join(train_root, aug)
                        sub_dataset = ImageFolder(path, transform= env_transform)
                        subset.append(sub_dataset)
                        print(f'Aug {environment} data path: {path}') 

                combined_dataset = ConcatDataset(subset)
                self.datasets.append(combined_dataset)
                 
            else:
                env_transform = transform
                path = os.path.join(test_root, environment)
                sub_dataset = ImageFolder(path, transform=env_transform)
                self.datasets.append(sub_dataset)
                # subset.append(sub_dataset)
                print(f'Test data path: {path}')    

        self.num_classes = len(sub_dataset.classes)

        self.input_shape = (3, 224, 224,)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "OfficeHome/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
            "region_4", "region_5"]
    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)


## Spawrious base classes
class CustomImageFolder(Dataset):
    """
    A class that takes one folder at a time and loads a set number of images in a folder and assigns them a specific class
    """
    def __init__(self, folder_path, class_index, limit=None, transform=None):
        self.folder_path = folder_path
        self.class_index = class_index
        self.image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        if limit:
            self.image_paths = self.image_paths[:limit]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        label = torch.tensor(self.class_index, dtype=torch.long)
        return img, label

class SpawriousBenchmark(MultipleDomainDataset):
    ENVIRONMENTS = ["Test", "SC_group_1", "SC_group_2"]
    input_shape = (3, 224, 224)
    num_classes = 4
    class_list = ["bulldog", "corgi", "dachshund", "labrador"]

    def __init__(self, train_combinations, test_combinations, root_dir, augment=True, type1=False):
        self.type1 = type1
        train_datasets, test_datasets = self._prepare_data_lists(train_combinations, test_combinations, root_dir, augment)
        self.datasets = [ConcatDataset(test_datasets)] + train_datasets

    # Prepares the train and test data lists by applying the necessary transformations.
    def _prepare_data_lists(self, train_combinations, test_combinations, root_dir, augment):
        test_transforms = transforms.Compose([
            transforms.Resize((self.input_shape[1], self.input_shape[2])),
            transforms.transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if augment:
            train_transforms = transforms.Compose([
                transforms.Resize((self.input_shape[1], self.input_shape[2])),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            train_transforms = test_transforms

        train_data_list = self._create_data_list(train_combinations, root_dir, train_transforms)
        test_data_list = self._create_data_list(test_combinations, root_dir, test_transforms)

        return train_data_list, test_data_list

    # Creates a list of datasets based on the given combinations and transformations.
    def _create_data_list(self, combinations, root_dir, transforms):
        data_list = []
        if isinstance(combinations, dict):
            
            # Build class groups for a given set of combinations, root directory, and transformations.
            for_each_class_group = []
            cg_index = 0
            for classes, comb_list in combinations.items():
                for_each_class_group.append([])
                for ind, location_limit in enumerate(comb_list):
                    if isinstance(location_limit, tuple):
                        location, limit = location_limit
                    else:
                        location, limit = location_limit, None
                    cg_data_list = []
                    for cls in classes:
                        path = os.path.join(root_dir, f"{0 if not self.type1 else ind}/{location}/{cls}")
                        data = CustomImageFolder(folder_path=path, class_index=self.class_list.index(cls), limit=limit, transform=transforms)
                        cg_data_list.append(data)
                    
                    for_each_class_group[cg_index].append(ConcatDataset(cg_data_list))
                cg_index += 1

            for group in range(len(for_each_class_group[0])):
                data_list.append(
                    ConcatDataset(
                        [for_each_class_group[k][group] for k in range(len(for_each_class_group))]
                    )
                )
        else:
            for location in combinations:
                path = os.path.join(root_dir, f"{0}/{location}/")
                data = ImageFolder(root=path, transform=transforms)
                data_list.append(data)

        return data_list
    
    
    # Buils combination dictionary for o2o datasets
    def build_type1_combination(self,group,test,filler):
        total = 3168
        counts = [int(0.97*total),int(0.87*total)]
        combinations = {}
        combinations['train_combinations'] = {
            ## correlated class
            ("bulldog",):[(group[0],counts[0]),(group[0],counts[1])],
            ("dachshund",):[(group[1],counts[0]),(group[1],counts[1])],
            ("labrador",):[(group[2],counts[0]),(group[2],counts[1])],
            ("corgi",):[(group[3],counts[0]),(group[3],counts[1])],
            ## filler
            ("bulldog","dachshund","labrador","corgi"):[(filler,total-counts[0]),(filler,total-counts[1])],
        }
        ## TEST
        combinations['test_combinations'] = {
            ("bulldog",):[test[0], test[0]],
            ("dachshund",):[test[1], test[1]],
            ("labrador",):[test[2], test[2]],
            ("corgi",):[test[3], test[3]],
        }
        return combinations

    # Buils combination dictionary for m2m datasets
    def build_type2_combination(self,group,test):
        total = 3168
        counts = [total,total]
        combinations = {}
        combinations['train_combinations'] = {
            ## correlated class
            ("bulldog",):[(group[0],counts[0]),(group[1],counts[1])],
            ("dachshund",):[(group[1],counts[0]),(group[0],counts[1])],
            ("labrador",):[(group[2],counts[0]),(group[3],counts[1])],
            ("corgi",):[(group[3],counts[0]),(group[2],counts[1])],
        }
        combinations['test_combinations'] = {
            ("bulldog",):[test[0], test[1]],
            ("dachshund",):[test[1], test[0]],
            ("labrador",):[test[2], test[3]],
            ("corgi",):[test[3], test[2]],
        }
        return combinations

## Spawrious classes for each Spawrious dataset 
class SpawriousO2O_easy(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ["desert","jungle","dirt","snow"]
        test = ["dirt","snow","desert","jungle"]
        filler = "beach"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'], type1=True)

class SpawriousO2O_medium(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['mountain', 'beach', 'dirt', 'jungle']
        test = ['jungle', 'dirt', 'beach', 'snow']
        filler = "desert"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'], type1=True)

class SpawriousO2O_hard(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['jungle', 'mountain', 'snow', 'desert']
        test = ['mountain', 'snow', 'desert', 'jungle']
        filler = "beach"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'], type1=True)

class SpawriousM2M_easy(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['desert', 'mountain', 'dirt', 'jungle']
        test = ['dirt', 'jungle', 'mountain', 'desert']
        combinations = self.build_type2_combination(group,test)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation']) 

class SpawriousM2M_medium(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['beach', 'snow', 'mountain', 'desert']
        test = ['desert', 'mountain', 'beach', 'snow']
        combinations = self.build_type2_combination(group,test)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'])
        
class SpawriousM2M_hard(SpawriousBenchmark):
    ENVIRONMENTS = ["Test","SC_group_1","SC_group_2"]
    def __init__(self, root_dir, test_envs, hparams):
        group = ["dirt","jungle","snow","beach"]
        test = ["snow","beach","dirt","jungle"]
        combinations = self.build_type2_combination(group,test)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'])