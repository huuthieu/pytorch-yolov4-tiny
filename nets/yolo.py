import torch
import torch.nn as nn

from nets.CSPdarknet53_tiny import darknet53_tiny
from nets.attention import cbam_block, eca_block, se_block

attention_block = [se_block, cbam_block, eca_block]

#-------------------------------------------------#  
#   Conv2d + BatchNormalization + LeakyReLU
#-------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

#---------------------------------------------------# 
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x


def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        BasicConv(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

#---------------------------------------------------#
#   trt_layer for export onnx and run tensorrt
#---------------------------------------------------#

class TrtYolo(nn.Module):
    def __init__(self, anchors, nc, input_shape):
        super(TrtYolo, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)
        self.nc = nc
        self.no = nc + 5
        self.nx, self.ny, self.ng = 0, 0, 0
        self.input_shape = input_shape
        self.training = False

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):
        bs, _, ny, nx = p.shape  # bs, 255, 13, 13
        stride_h = self.input_shape[0] / ny
        stride_w = self.input_shape[1] / nx
        self.anchor_vec = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors]
        self.anchor_wh = self.anchor_vec.view(1,self.na, 1,1,2)
        if (self.nx, self.ny) != (nx, ny):
            self.create_grids((nx, ny), p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction
        io = p.clone()  # inference output
        io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
        io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
        _scale = torch.Tensor([nx, ny, nx, ny]).type(FloatTensor)
        io[...,:4] /= _scale
        # torch.sigmoid_(io[..., 4:])
        io[...,4:5] = torch.sigmoid(io[..., 4:5])
        return io.view(bs, -1, self.no)  # view [1, 3, 13, 13, 85] as [1, 507, 85]

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#

class YoloBody(nn.Module):
    def __init__(self, anchors, anchors_mask, num_classes, phi=0, EXPORT_TRT = False):
        super(YoloBody, self).__init__()
        self.phi            = phi
        self.backbone       = darknet53_tiny(None)

        self.conv_for_P5    = BasicConv(512,256,1)
        self.yolo_headP5    = yolo_head([512, len(anchors_mask[0]) * (5 + num_classes)],256)

        self.upsample       = Upsample(256,128)
        self.yolo_headP4    = yolo_head([256, len(anchors_mask[1]) * (5 + num_classes)],384)

        if 1 <= self.phi and self.phi <= 3:
            self.feat1_att      = attention_block[self.phi - 1](256)
            self.feat2_att      = attention_block[self.phi - 1](512)
            self.upsample_att   = attention_block[self.phi - 1](128)

        self.anchors = anchors
        self.trt_export = EXPORT_TRT

    def forward(self, x):
        #---------------------------------------------------#
        #   CSPdarknet53_tiny backbone
        #   feat1 shape: 26,26,256
        #   feat2 shape: 13,13,512
        #---------------------------------------------------#
        feat1, feat2 = self.backbone(x)
        if 1 <= self.phi and self.phi <= 3:
            feat1 = self.feat1_att(feat1)
            feat2 = self.feat2_att(feat2)

        # 13,13,512 -> 13,13,256
        P5 = self.conv_for_P5(feat2)
        # 13,13,256 -> 13,13,512 -> 13,13,255
        if self.trt_export and not self.training:
            out0 = self.yolo_headP5(P5)
            out0 = TrtYolo(self.anchors[[3,4,5]],1,32)(out0)
        else:
            out0 = self.yolo_headP5(P5)
        # 13,13,256 -> 13,13,128 -> 26,26,128
        P5_Upsample = self.upsample(P5)
        # 26,26,256 + 26,26,128 -> 26,26,384
        if 1 <= self.phi and self.phi <= 3:
            P5_Upsample = self.upsample_att(P5_Upsample)
        P4 = torch.cat([P5_Upsample,feat1],axis=1)

        # 26,26,384 -> 26,26,256 -> 26,26,255
        if self.trt_export and not self.training:
            out1 = self.yolo_headP4(P4)
            out1 = TrtYolo(self.anchors[[0,1,2]],1,16)(out1)
        else:
            out1 = self.yolo_headP4(P4)

        return out0, out1
