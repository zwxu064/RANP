from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import convbn, convbn_3d, feature_extraction, disparityregression

class hourglass(nn.Module):
    def __init__(self, inplanes, activation_mode='ReLU', upsample_mode='transpose'):
        super(hourglass, self).__init__()
        activation = nn.ReLU(inplace=True) if (activation_mode == 'ReLU') else nn.LeakyReLU(0.2, inplace=True)
        self.activation = activation
        self.upsample_mode = upsample_mode

        # Group 1
        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   activation)
        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

        # Group 2
        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   activation)
        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   activation)

        # Group 3
        if upsample_mode == 'transpose':
          self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                     nn.BatchNorm3d(inplanes*2)) #+conv2

          self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                     nn.BatchNorm3d(inplanes)) #+x
        elif upsample_mode == 'interpolation_conv':
          # self.conv5 = nn.Sequential(nn.Conv3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, padding=1, bias=False),
          #                            nn.BatchNorm3d(inplanes*2))
          self.conv5 = nn.Sequential(nn.BatchNorm3d(inplanes*2))
          self.conv6 = nn.Sequential(nn.Conv3d(inplanes*2, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm3d(inplanes))
          # self.conv6 = nn.Sequential(nn.BatchNorm3d(inplanes))

    def forward(self, x, presqu, postsqu):
        out = self.conv1(x) #in:1/4 out:1/8
        pre = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None:
           pre = self.activation(pre + postsqu)
        else:
           pre = self.activation(pre)

        out = self.conv3(pre) #in:1/8 out:1/16
        out = self.conv4(out) #in:1/16 out:1/16

        if self.upsample_mode == 'transpose':
          if presqu is not None:
            post = self.activation(self.conv5(out) + presqu) #in:1/16 out:1/8
          else:
            post = self.activation(self.conv5(out) + pre)

          out = self.conv6(post)  # in:1/8 out:1/4
        elif self.upsample_mode == 'interpolation_conv':
          d, h, w = out.shape[-3:]
          out = F.interpolate(out, (2*d, 2*h, 2*w), mode='trilinear', align_corners=True)
          if presqu is not None:
            post = self.activation(self.conv5(out) + presqu)
          else:
            post = self.activation(self.conv5(out) + pre)

          d, h, w = post.shape[-3:]
          post_2x = F.interpolate(post, (2*d, 2*h, 2*w), mode='trilinear', align_corners=True)
          out = self.conv6(post_2x)

        return out, pre, post

class PSMNet(nn.Module):
    def __init__(self, maxdisp, activation_mode='ReLU', enable_out_list=False, reduce_glass=0, cv_type=None):
        super(PSMNet, self).__init__()
        activation = nn.ReLU(inplace=True) if (activation_mode == 'ReLU') else nn.LeakyReLU(0.2, inplace=True)
        self.enable_out_list = enable_out_list
        self.maxdisp = maxdisp
        self.reduce_glass = reduce_glass
        self.cv_type = cv_type
        cv_in = 32 if (self.cv_type == 'abs_diff') else 64
        self.feature_extraction = feature_extraction(activation_mode)

        self.dres0 = nn.Sequential(convbn_3d(cv_in, 32, 3, 1, 1),
                                   activation,
                                   convbn_3d(32, 32, 3, 1, 1),
                                   activation)

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   activation,
                                   convbn_3d(32, 32, 3, 1, 1)) 

        if reduce_glass <= 2:
          self.dres2 = hourglass(32)
          self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                        activation,
                                        nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        if reduce_glass <= 1:
          self.dres3 = hourglass(32)
          self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                        activation,
                                        nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        if reduce_glass <= 0:
          self.dres4 = hourglass(32)
          self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                        activation,
                                        nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right, edge_weights=None):
        # Zhiwei left: (1,3,256,512)->down to 1/4, refimg_fea: (1,32,64,128)
        refimg_fea, _ = self.feature_extraction(left)
        targetimg_fea, _ = self.feature_extraction(right)

        #matching
        if self.cv_type == 'abs_diff':
          cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1],
                                            self.maxdisp // 4, refimg_fea.size()[2],
                                            refimg_fea.size()[3]).zero_()).cuda()
          for i in range(self.maxdisp // 4):
            if i > 0:
              cost[:, :, i, :, i:] = (refimg_fea[:, :, :, i:] - targetimg_fea[:, :, :, :-i]).abs()
            else:
              cost[:, :, i, :, :] = (refimg_fea - targetimg_fea).abs()
        else:
          cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2,
                                            self.maxdisp//4,  refimg_fea.size()[2],
                                            refimg_fea.size()[3]).zero_()).cuda()
          for i in range(self.maxdisp//4):
              if i > 0 :
               cost[:, :refimg_fea.size()[1], i, :, i:]   = refimg_fea[:,:,:,i:]
               cost[:, refimg_fea.size()[1]:, i, :, i:] = targetimg_fea[:,:,:,:-i]
              else:
               cost[:, :refimg_fea.size()[1], i, :, :]   = refimg_fea
               cost[:, refimg_fea.size()[1]:, i, :, :]   = targetimg_fea

        cost = cost.contiguous()

        # Zhiwei cost: (1,64,48,64,128)

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        if self.reduce_glass <= 2:
          out1, pre1, post1 = self.dres2(cost0, None, None)
          out1 = out1+cost0
          cost1 = self.classif1(out1)

        if self.reduce_glass <= 1:
          out2, pre2, post2 = self.dres3(out1, pre1, post1)
          out2 = out2+cost0
          cost2 = self.classif2(out2) + cost1

        if self.reduce_glass <= 0:
          out3, pre3, post3 = self.dres4(out2, pre1, post2)
          out3 = out3+cost0
          cost3 = self.classif3(out3) + cost2

        if self.training:
          # Zhiwei From (1,1,48,64,128)->up to 4, (1,1,192,256,512)
          if self.reduce_glass <= 2:
            cost1 = F.interpolate(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear', align_corners=True)
            cost1 = torch.squeeze(cost1,1)
            pred1 = F.softmax(cost1,dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)

          if self.reduce_glass <= 1:
            cost2 = F.interpolate(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear', align_corners=True)
            cost2 = torch.squeeze(cost2,1)
            pred2 = F.softmax(cost2,dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)

          if self.reduce_glass <= 0:
            cost3 = F.interpolate(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear', align_corners=True)
            cost3 = torch.squeeze(cost3,1)
            pred3 = F.softmax(cost3,dim=1)
            #For your information: This formulation 'softmax(c)' learned "similarity"
            #while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
            #However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
            pred3 = disparityregression(self.maxdisp)(pred3)
        else:
          if self.reduce_glass == 2:
            cost1 = F.interpolate(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear', align_corners=True)
            cost1 = torch.squeeze(cost1,1)
            pred1 = F.softmax(cost1,dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)
          elif self.reduce_glass == 1:
            cost2 = F.interpolate(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear', align_corners=True)
            cost2 = torch.squeeze(cost2,1)
            pred2 = F.softmax(cost2,dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)
          elif self.reduce_glass == 0:
            cost3 = F.interpolate(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear', align_corners=True)
            cost3 = torch.squeeze(cost3,1)
            pred3 = F.softmax(cost3,dim=1)
            pred3 = disparityregression(self.maxdisp)(pred3)

        if self.reduce_glass == 0:
          if self.training:
            if self.enable_out_list:
              return [pred1, pred2, pred3], None
            else:
              return pred1, pred2, pred3
          else:
              return pred3, None
        elif self.reduce_glass == 1:
          if self.training:
            if self.enable_out_list:
              return [pred1, pred2], None
            else:
              return pred1, pred2
          else:
            return pred2, None
        elif self.reduce_glass == 2:
          if self.training:
            if self.enable_out_list:
              return [pred1], None
            else:
              return pred1
          else:
            return pred1, None
