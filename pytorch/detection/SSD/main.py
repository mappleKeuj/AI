
import os
import torch

from torch.autograd import Variable

from ssd.model import SSD300, Loss
from ssd.utils import dboxes300_coco, Encoder, tencent_trick, calc_iou_tensor
from ssd.data import get_dataset, get_dataloader


if __name__ == "__main__":
    
    epochs = 300
    data_root = r"C:\Users\delfr\fiftyone\open-images-v6"
    phases = ["train", "validation"]
    device = "cuda"
    
    batch_size = 32
    shuffle = True
    num_workers = 4
    
    ssd_model = SSD300(backbone="resnet18")
    ssd_model = ssd_model.to(device)

    # scaler
    scaler = torch.cuda.amp.GradScaler()
    # optimizer
    optimizer = torch.optim.SGD(tencent_trick(ssd_model), lr=2.6e-3, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.SGD(ssd_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)

    # Setup data, defaults
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes=dboxes)
    loss_func = Loss(dboxes).to(device)

    # create dataloaderts
    labels = ["/m/01jfm_"] # "Vehicle registration plate"
    dataloaders = {}
    for phase in phases:
        # create training set    
        path = os.path.join(data_root, phase, "data")
        annotation = os.path.join(data_root, phase, "labels", "detections.csv")
        dataset = get_dataset(path, annotation, labels)
        dataloader = get_dataloader(dataset, batch_size, shuffle, num_workers)
        print(f"Dataloader created for phase {phase} - nb images {len(dataset)} - nb batches {len(dataloader)}")
        dataloaders[phase] = dataloader
        
    for epoch in range(1, epochs + 1):
        for phase in phases:
            print(f"Epoch: {epoch} - Phase {phase}")
            if phase == "train":
                ssd_model = ssd_model.train()
            else:
                ssd_model = ssd_model.eval()
            
            running_loss = 0.0    
            for nbatch, (img, img_id, img_size, bbox_sizes, bbox_labels) in enumerate(dataloaders[phase]):
                img = img.to(device)
                bbox_labels = bbox_labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                if phase == "train":                    
                    with torch.cuda.amp.autocast():
                        loc, pred = ssd_model(img)
                        gloc = Variable(bbox_sizes, requires_grad=False)
                        glabel = Variable(bbox_labels, requires_grad=False)

                        trans_bbox = gloc.transpose(1, 2).contiguous().to(device)
                        loss = loss_func(loc, pred, trans_bbox, glabel)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer=optimizer)
                    scaler.update()
                    running_loss += loss.item()   
                else:
                    loc, pred = ssd_model(img)
                    gloc = Variable(bbox_sizes, requires_grad=False)
                    glabel = Variable(bbox_labels, requires_grad=False)

                    trans_bbox = gloc.transpose(1, 2).contiguous().to(device)
                    running_loss += loss_func(loc, pred, trans_bbox, glabel).item()
                                   
                    # for idx in range(loc.shape[0]):               
                    #     ploc_i = loc[idx, :, :].unsqueeze(0)
                    #     plabel_i = pred[idx, :, :].unsqueeze(0)
                        
                    #     result = encoder.decode_batch(ploc_i, plabel_i, 0.50, 200)[0]
                    #     if result[0].nelement() != 0:
                    #         print(0)  
                        
                                   
                    # for i, bb in enumerate(loc):
                    #     iou = calc_iou_tensor(bb, trans_bbox[i])
                    #     print(iou)
                        
            print(f"mean loss {running_loss / len(dataloaders[phase])}")
        



