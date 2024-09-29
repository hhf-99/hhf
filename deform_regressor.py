import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.deform_conv import DeformConv2d

class deform_fusion(nn.Module):
    def __init__(self, opt, in_channels=768*5, cnn_channels=256*3, out_channels=256*3):
        #in_channels表示输入通道的数量（默认为768 * 5），
        #cnn_channels表示卷积层的输出通道数量（默认为256 * 3），out_channels表示网络最终输出的通道数量（默认为256 * 3）
        super().__init__()
        #in_channels, out_channels, kernel_size, stride, padding
        self.d_hidn = 512    #隐藏单元的数量
        if opt.patch_size == 8:
            stride = 1
        else:
            stride = 2
        self.conv_offset = nn.Conv2d(in_channels, 2*3*3, 3, 1, 1)
        #接受in_channels个输入通道，有2*3*3个输出通道，使用3x3的卷积核，步幅为1，填充为1。
        #生成可变形卷积（Deformable Convolution）的偏移量。
        self.deform = DeformConv2d(cnn_channels, out_channels, 3, 1, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=self.d_hidn, kernel_size=3,padding=1,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.d_hidn, out_channels=out_channels, kernel_size=3, padding=1,stride=stride)
        )#将特征图的尺寸调整为期望的大小，并输出最终的特征表示。
#定义了一个名为conv1的序列模块，它包含了一系列的卷积层和激活函数。
    def forward(self, cnn_feat, vit_feat):
        vit_feat = F.interpolate(vit_feat, size=cnn_feat.shape[-2:], mode="nearest")
        #使用F.interpolate函数将vit_feat插值（或调整）到与cnn_feat具有相同的尺寸。
        # 插值方法使用最近邻插值(nearest)，以确保它们的尺寸相匹配。
        offset = self.conv_offset(vit_feat)
        deform_feat = self.deform(cnn_feat, offset)
        deform_feat = self.conv1(deform_feat)
        
        return deform_feat

class Pixel_Prediction(nn.Module):
    def __init__(self, inchannels=768*5+256*3, outchannels=256, d_hidn=1024):
        super().__init__()
        self.d_hidn = d_hidn
        self.down_channel = nn.Conv2d(inchannels, outchannels, kernel_size=1)
        self.feat_smoothing = nn.Sequential(
            nn.Conv2d(in_channels=256*3, out_channels=self.d_hidn, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.d_hidn, out_channels=512, kernel_size=3, padding=1)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3,padding=1), 
            nn.ReLU()
        )
        self.conv_attent =  nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
        )

        self.fc1_q = nn.Linear(512, 256)
        self.fc2_q = nn.Linear(256, 1)
        ##
        self.dropout = nn.Dropout()
        ##
    def forward(self,f_eha, f_ref, cnn_eha, cnn_ref):
        # f_eha = torch.cat((f_eha,cnn_eha),1)
        # f_ref = torch.cat((f_ref,cnn_ref),1)
        # f_eha = self.down_channel(f_eha)
        # f_ref = self.down_channel(f_ref)
        #
        # f_cat = torch.cat((f_eha - f_ref, f_eha, f_ref), 1)
        #
        # feat_fused = self.feat_smoothing(f_cat)
        # feat = self.conv1(feat_fused)
        # f = self.conv(feat)
        # w = self.conv_attent(feat)
        # pred = (f*w).sum(dim=2).sum(dim=2)/w.sum(dim=2).sum(dim=2)
        #
        # return pred
        f_eha = torch.cat((f_eha, cnn_eha), 1)
        f_ref = torch.cat((f_ref, cnn_ref), 1)
        f_eha = self.down_channel(f_eha)
        f_ref = self.down_channel(f_ref)
        f_cat = torch.cat((f_eha - f_ref, f_eha, f_ref), 1)

        feat_fused = self.feat_smoothing(f_cat)

        ##
        b, c, h, w = feat_fused.size()
        feat_fused = feat_fused.view(-1, 512)
        ##

        f1 = F.relu(self.fc1_q(feat_fused))

        f2 = self.dropout(f1)
        f = self.fc2_q(f2)
        ##
        f1 = f1.view(b, 256, h, w)
        f = f.view(b, 1, h, w)
        ##
        w = self.conv_attent(f1)
        # pred = torch.sum(f * w) / torch.sum(w)
        pred = (f * w).sum(dim=2).sum(dim=2) / w.sum(dim=2).sum(dim=2)
        return pred