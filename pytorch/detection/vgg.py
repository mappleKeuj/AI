

import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, config_name="vgg_11", in_channels=3, num_classes=1000) -> None:
        super(VGG, self).__init__()
        self.features = self._create_features_layers(config_name, in_channels=in_channels)
        self.avgPooling = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.flatten = nn.Flatten(start_dim=0, end_dim=-1)
        self.classifier = self._create_classifier_layer(num_classes=num_classes)
    
    def _create_features_layers(self, vgg_config, in_channels):
        net_config, use_batch_norm = vgg_config.get_config()

        node_list=[]
        in_dim = in_channels
        for dim in net_config:
            if dim == "M":
                node_list.append(
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                    )
            else:
                node_list.append(
                        nn.Conv2d(in_channels=in_dim, out_channels=dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    )
                if use_batch_norm:
                    node_list.append(
                            nn.BatchNorm2d(num_features=dim, eps=1e-05, affine=True, track_running_stats=True)
                        )
                node_list.append(nn.ReLU(inplace=True))
                in_dim = dim        
                
        return nn.Sequential(*node_list)
    
    def _create_classifier_layer(self, num_classes):
        classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=0)
        )
        return classifier
         
    def forward(self, x):
        x = self.features(x)
        x = self.avgPooling(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x