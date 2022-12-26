
import torch

def train_loop(model, epochs, criterion, optimizer, device, train_dl, val_dl, logging):
    """
    Support function for model training.

    Args:
        model: Model to be trained
        num_epochs: Number of epochs
        criterion: Optimization criterion (loss)
        optimizer: Optimizer to use for training
        device: Device to run the training on. Must be 'cpu' or 'cuda'
        train_dl: Dataloader for training dataset
        val_dl: Dataloader for validation dataset
        logging: Logger obj
    """
    
    # Move model to GPU
    model = model.to(device)

    # Each epoch has a training and validation phase
    phases = ["train", "val"]
    dataloader = dict(zip(phases, [train_dl, val_dl]))

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    for epoch in range(epochs):
        for phase in phases:
            if phase == "train":
                model.train()
            else:
                model.eval()
            
            logging.info(f"Run Epoch {epoch} for phase: {phase}")
            
            running_loss = 0.0
            running_corrects = 0
            visited = 0    
    
            # Iterate over data.
            for imgs, labels in dataloader[phase]:
                imgs = imgs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(imgs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                
                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                    
                # statistics
                visited += imgs.size(0)
                running_loss += loss.item() * imgs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
  
            epoch_loss = running_loss / visited
            epoch_acc = running_corrects.double() / visited
            
            logging.info(
                f"{visited} images seen, \n \
                Epoch Loss {epoch_loss}, \n\
                Epoch Accuracy {epoch_acc}"
            )
            
    
