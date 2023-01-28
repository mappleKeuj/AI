
import torch 
import logging

from vgg import VGG
from vgg.train import train_loop
from vgg.config import VggConfig

from vgg.datasets import create_dataset_loader

from torch.utils.data import random_split
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import InterpolationMode


if __name__ == '__main__': 
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )

    transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=(0, 360), interpolation=InterpolationMode.BILINEAR, expand=True),
        T.Resize((300, 300)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = "C:\Programming\computer_vision\datasets\classification\FastFood"
    logging.info(f"Create dataset from {data_dir}")
    dataset = ImageFolder(data_dir, transforms)
    num_classes = len(dataset.classes)

    train_ds, val_ds, test_ds =  random_split(dataset=dataset, lengths=[0.9, 0.1, 0.0])
    logging.info(
            f"Total number of images {len(dataset)},\n \
            Lenght train_ds: {len(train_ds)},\n \
            Lenght val_ds: {len(val_ds)},\n \
            Lenght test_ds: {len(test_ds)}, \n \
            Number of Classes: {num_classes}"
        )

    train_dl = create_dataset_loader(dataset=train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = create_dataset_loader(dataset=val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    logging.info(f"Use device: {device}")

    # config_name = VggConfig().get_list_of_configs()[3]
    # vgg_config = VggConfig(config_name=config_name, use_batch_norm=True)

    # logging.info(f"Create vgg with config: {repr(vgg_config)}")
    # vgg_model = VGG(vgg_config, in_channels=3, num_classes=num_classes)
    # print(vgg_model)
    
    vgg_model = torch.hub.load("pytorch/vision:v0.10.0", "vgg11", pretrained=True)
    vgg_model.classifier[6] = torch.nn.Linear(4096, num_classes, True)
    print(vgg_model)
    
    logging.info("Start training loop")
    epochs=300
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(vgg_model.parameters(), lr=0.001, momentum=0.9) 

    train_loop(
            model=vgg_model, 
            epochs=epochs, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=device,
            train_dl=train_dl,
            val_dl=val_dl,
            logging=logging
        )

