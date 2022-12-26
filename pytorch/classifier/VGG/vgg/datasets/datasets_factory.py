
from torch.utils.data import DataLoader

def create_dataset_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)