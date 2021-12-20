class ImageDataLoader(DataLoader):
    """
    Load Image datasets from torch
    Default category is MNIST
    Pass category as 'MNIST' or 'CIFAR10'
    """

    def __init__(self, train_transforms, test_transforms, data_dir, batch_size, shuffle, category='custom',
                 num_workers=4, pin_memory=False, device='cpu',
                 figure_size=(20, 8), test_pct=0.1):
        self.data_dir = data_dir
        self.figure_size = figure_size
        cuda = torch.cuda.is_available()
        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': pin_memory
        }

        def get_augmentation(transforms):
            return lambda img: transforms(image=np.array(img))['image']

        if cuda:
            self.device = 'gpu'
            pin_memory = True
        else:
            self.device = device

        self.classes = None

        
        self.train_loader = datasets.CIFAR10(
          self.data_dir,
          train=True,
          download=True,
          transform=get_augmentation(train_transforms)
          # transform=transforms.build_transforms(train=True)
            )
        self.test_loader = datasets.CIFAR10(
          self.data_dir,
          train=False,
          download=True,
          # transform=transforms.build_transforms(train=False)
          transform=get_augmentation(test_transforms)
	    )
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
                'horse', 'ship', 'truck')
        self.train_loader = DataLoader(self.train_loader, shuffle=shuffle, **self.init_kwargs)
        self.test_loader = DataLoader(self.test_loader, **self.init_kwargs)