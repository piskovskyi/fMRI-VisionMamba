import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse
from models import build_vssm_model
from config import get_config

def get_vmamba(num_lh, num_rh):
    checkpoint_path = "vssm_tiny_0230_ckpt_epoch_262.pth"
    config_path = "vssm_tiny_224_0229flex.yaml"
    
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', default = config_path, 
                        type=str, required=False, metavar="FILE", help='path to config file', )
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+', )
    argsVMamba, unparsed = parser.parse_known_args()

    config = get_config(argsVMamba)

    model = build_vssm_model(config, num_lh = num_lh, num_rh = num_rh, is_pretrain=True)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)

    return model

class Model(nn.Module):
    def __init__(self, num_lh, num_rh):
        super(Model, self).__init__()

        # 28x28x1 => 26x26x32
        self.backbone = get_vmamba()
        self.backbone.train()

        for p in self.backbone.parameters():
            p.requires_grad = True


        self.fcLeft = nn.Linear(1000, num_lh)
        self.fcRight = nn.Linear(1000, num_rh)

    def forward(self, x):
        # NumImages x 224x224x3 => NumImages x 1000
        x = self.backbone(x)

        x = F.relu(x)

        xL = self.fcLeft(x)
        xR = self.fcRight(x)

        return xL, xR
   


class Model2Layers(nn.Module):
    def __init__(self, num_lh, num_rh):
        super(Model2Layers, self).__init__()

        # 28x28x1 => 26x26x32
        self.backbone = get_vmamba()
        self.backbone.train()

        for p in self.backbone.parameters():
            p.requires_grad = True
        

        self.fcLeft1 = nn.Linear(1000, 7000) 
        self.fcRight1 = nn.Linear(1000, 7000) 

        self.fcLeft2 = nn.Linear(7000, num_lh) 
        self.fcRight2 = nn.Linear(7000, num_rh) 


    def forward(self, x):
        # NumImages x 3 x 224 x 224 => NumImages x 1000
        x = self.backbone(x)

        x = F.relu(x) 

        # NumImages x 1000 => NumImages x 5000 
        xL = self.fcLeft1(x)
        xR = self.fcRight1(x)

        xL = F.relu(xL) 
        xR = F.relu(xR) 
        
        xL = self.fcLeft2(xL) 
        xR = self.fcRight2(xR) 

        return xL, xR


class ModelNoClassifier(nn.Module):
    def __init__(self, num_lh, num_rh):
        super(ModelNoClassifier, self).__init__()

        # 28x28x1 => 26x26x32
        self.backbone = get_vmamba(num_lh, num_rh)
        self.backbone.train()

        for p in self.backbone.parameters():
            p.requires_grad = True


    def forward(self, x):
        xL, xR = self.backbone(x)

        return xL, xR
    

class ModelNoClassifier2Layers(nn.Module):
    def __init__(self, num_lh, num_rh):
        super(ModelNoClassifier2Layers, self).__init__()

        # 28x28x1 => 26x26x32
        self.backbone = get_vmamba()
        self.backbone.train()

        for p in self.backbone.parameters():
            p.requires_grad = True
        
        self.fcLeft = nn.Linear(5000, num_lh)
        self.fcRight = nn.Linear(5000, num_rh)


    def forward(self, x):
        xL, xR = self.backbone(x)

        xL = F.relu(xL)
        xR = F.relu(xR)


        xL = self.fcLeft(xL) 
        xR = self.fcRight(xR) 


        return xL, xR