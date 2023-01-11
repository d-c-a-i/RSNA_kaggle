import sys
sys.path.append('/home/derek/Desktop/RSNA_baseline_kaggle/src')
import torch
from torch import nn
import timm

def conv1x1(in_channel, out_channel): #not change resolution
    return nn.Conv2d(in_channel,out_channel,
                      kernel_size=1,stride=1,padding=0,dilation=1,bias=False)

def conv3x3(in_channel, out_channel): #not change resolusion
    return nn.Conv2d(in_channel,out_channel,
                      kernel_size=3,stride=1,padding=1,dilation=1,bias=False)

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.xavier_uniform_(m.weight, gain=1)
        #nn.init.xavier_normal_(m.weight, gain=1)
        #nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Batch') != -1:
        m.weight.data.normal_(1,0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Embedding') != -1:
        nn.init.orthogonal_(m.weight, gain=1)

#hypercolumns + deepsupervision
class Custom_effnetv2_s(nn.Module):
    def __init__(self, last_k_layers, IMAGENET_pretrained=True, drop_rate=0.0, drop_path_rate =0.0):
        super().__init__()
        self.last_k_layers = last_k_layers
        assert 1<= self.last_k_layers <= 6
        if IMAGENET_pretrained:
            effnetv2_s = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=0, drop_rate=0.0, drop_path_rate =0.0)
        else:
            effnetv2_s = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=0, drop_rate=0.0, drop_path_rate =0.0)
            
        self.encoder0 = nn.Sequential(
            effnetv2_s.conv_stem,
            effnetv2_s.bn1,
            effnetv2_s.blocks[0]
        ) #->(*,24,h/2,w/2)
        self.encoder1 = effnetv2_s.blocks[1] #->(*,48,h/4,w/4)
        self.encoder2 = effnetv2_s.blocks[2] #->(*,64,h/8,w/8)
        self.encoder3 = effnetv2_s.blocks[3] #->(*,128,h/16,w/16)
        self.encoder4 = effnetv2_s.blocks[4] #->(*,160,h/16,w/16)
        self.encoder5 = effnetv2_s.blocks[5] #->(*,256,h/32,w/32)
        
        
        self.deep5 = conv1x1(256,1).apply(init_weight)
        self.deep4 = conv1x1(160,1).apply(init_weight)
        self.deep3 = conv1x1(128,1).apply(init_weight)
        self.deep2 = conv1x1(64,1).apply(init_weight)
        self.deep1 = conv1x1(48,1).apply(init_weight)
        self.deep0 = conv1x1(24,1).apply(init_weight)
        
        
        self.upsample5 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.clf = nn.Sequential(
            nn.BatchNorm1d(256).apply(init_weight),
            nn.Linear(256,64).apply(init_weight),
            nn.ELU(True),
            nn.BatchNorm1d(64).apply(init_weight),
            nn.Linear(64,1).apply(init_weight)
        )
        
        hypercol_channel_num = [-100, 256, 256+160, 256+160+128, 256+160+128+64, 256+160+128+64+48, 256+160+128+64+48+24][self.last_k_layers]
        assert hypercol_channel_num != 100
        
        
        # self.final_conv = nn.Sequential(
        #     conv3x3(hypercol_channel_num, hypercol_channel_num//8).apply(init_weight),
        #     nn.ELU(True),
        #     conv1x1(hypercol_channel_num//8, 1).apply(init_weight)
        # )
    
        # consider using the same clf as above??
        self.hypercol_clf = nn.Sequential(
            nn.BatchNorm1d(hypercol_channel_num).apply(init_weight),
            nn.Linear(hypercol_channel_num, hypercol_channel_num//4).apply(init_weight),
            nn.ELU(True),
            nn.BatchNorm1d(hypercol_channel_num//4).apply(init_weight),
            nn.Linear(hypercol_channel_num//4,1).apply(init_weight)
        )
        
        deep_channel_num = [-100, 1, 2, 3, 4, 5, 6][self.last_k_layers]
        assert deep_channel_num != 100
        self.deep_clf = nn.Sequential(
            nn.BatchNorm1d(deep_channel_num).apply(init_weight),
            nn.Linear(deep_channel_num, 2).apply(init_weight),
            nn.ELU(True),
            nn.BatchNorm1d(2).apply(init_weight),
            nn.Linear(2,1).apply(init_weight)
        )
        
    def forward(self, input):
        x0 = self.encoder0(input) #->(*,24,h/2,w/2)
        x1 = self.encoder1(x0) #->(*,48,h/4,w/4)
        x2 = self.encoder2(x1) #->(*,64,h/8,w/8)
        x3 = self.encoder3(x2) #->(*,128,h/16,w/16)
        x4 = self.encoder4(x3) #->(*,160,h/16,w/16)
        x5 = self.encoder5(x4) #->(*,256,h/32,w/32)
        
        logits_clf = self.clf(self.avgpool(x5).squeeze(-1).squeeze(-1))
        
        # y5 = self.upsample5(x5) #->(*,256,h,w)
        # y4 = self.upsample4(x4) #->(*,160,h,w)
        # y3 = self.upsample3(x3) #->(*,128,h,w)
        # y2 = self.upsample2(x2) #->(*,64,h,w)
        # y1 = self.upsample1(x1) #->(*,48,h,w)
        # y0 = self.upsample0(x0) #->(*,24,h,w)
        
        # hypercol_list = [y5,y4,y3,y2,y1,y0][:self.last_k_layers]
        # hypercol = torch.cat(hypercol_list, dim=1)
        
        # logits_hypercol = self.hypercol_clf(self.avgpool(hypercol).squeeze(-1).squeeze(-1))
        
        
        # s5 = self.deep5(y5)
        # s4 = self.deep4(y4)
        # s3 = self.deep3(y3)
        # s2 = self.deep2(y2)
        # s1 = self.deep1(y1)
        # s0 = self.deep0(y0)
        # deeps_list = [s5,s4,s3,s2,s1,s0][:self.last_k_layers]
        # deeps = torch.cat(deeps_list, dim=1)
        
        # logits_deeps = self.deep_clf(self.avgpool(deeps).squeeze(-1).squeeze(-1))
            
            
        # return logits_clf, logits_deeps, logits_hypercol
        return logits_clf.squeeze()




# if __name__ == "__main__":
#     # import timm
#     # from torchsummary import summary
#     # model = timm.create_model('seresnext101_32x4d', pretrained=True, num_classes=0)
#     # print(summary(model.cuda(), (3, 224, 224)))
#     model = Custom_effnetv2_s(resolution=(1024,512), deepsupervision=True)
#     model(torch.randn(2, 3, 1024, 512))
