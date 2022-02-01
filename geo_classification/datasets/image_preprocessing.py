import torchvision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def tfms(dataset_type: str, image_size=224):

    if dataset_type == "train":
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomResizedCrop(image_size, scale=(0.66, 1.0)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    elif dataset_type == "valid":
        if image_size < 250:
            return torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(256),
                    torchvision.transforms.CenterCrop(image_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.CenterCrop(image_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    raise KeyError
