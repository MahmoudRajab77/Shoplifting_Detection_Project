#-----------------------------< Imports >-----------------------------
import torch
import torch.nn as nn
import torchvision.models.video as video_models






#----------------------------------------------------------------< Functions >-----------------------------------------------------------------
class PretrainedR3D(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=False):
        super(PretrainedR3D, self).__init__()
        
        self.model = video_models.r3d_18(pretrained=True)
        
        in_features = self.model.fc.in_features
        
        self.model.fc = nn.Linear(in_features, num_classes)
        
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            
            for param in self.model.fc.parameters():
                param.requires_grad = True
    #-------------------------------------------------------
    def forward(self, x):
        return self.model(x)
      
