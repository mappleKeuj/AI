
from abc import abstractmethod

class VggConfig():
    def __init__(self, config_name = "vgg_11", use_batch_norm=True):
        self.config_name = config_name
        self.use_batch_norm = use_batch_norm
    
    @abstractmethod 
    def get_list_of_configs(self):
        return ["vgg_11", "vgg_13", "vgg_16", "vgg_19"]

    def get_config(self):
        if "vgg_11" in self.config_name:
            config = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
            return config, self.use_batch_norm
        
    def __repr__(self):
        repr = f"VggConfig(): {self.config_name} with batch norm. {self.use_batch_norm}"
        return repr
