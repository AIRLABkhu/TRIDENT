class target_augmented_PACS(MultipleDomainDataset):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs,  hparams):
        super().__init__()
        augment = hparams['data_augmentation']
        train_root = os.path.join(root, hparams['data_augmentation_root'])
        test_root = os.path.join(root, 'PACS')
        dirs = os.listdir(test_root)
        dirs = sorted(dirs)
        test_dirs = [dirs[x] for x in test_envs]
        train_envs = [x for x in range(4) if x not in test_envs]

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
                content_dir = os.path.join(train_root, f'PACS_{environment}')
                style = glob.glob(os.path.join(train_root, f"*/{environment}/"))
                for sty in style:
                    sub_dataset = ImageFolder(sty, transform= env_transform)
                    subset.append(sub_dataset)
                    print(f'Train data path: {sty}')
                path = os.path.join(test_root, environment)
                print(f'Train data path: {path}')
                sub_dataset = ImageFolder(path, transform=env_transform)
                subset.append(sub_dataset)
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

