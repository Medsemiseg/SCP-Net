# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
import torch.nn.functional as F
import numpy as np
class ConvBlock_2d(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock_2d, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv2d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""
    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, mode_upsampling=1):
        super(UpBlock, self).__init__()
        self.mode_upsampling = mode_upsampling
        if mode_upsampling==0:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        elif mode_upsampling==1:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif mode_upsampling==2:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        elif mode_upsampling==3:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.mode_upsampling != 0:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]
class DecoderCCT(nn.Module):
    def __init__(self, params):
        super(DecoderCCT, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output
class Decoder_sdf(nn.Module):
    def __init__(self, params):
        super(Decoder_sdf, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, mode_upsampling=self.up_type)
        self.out_conv2 = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout2d(p=0.5, inplace=False)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        outputsdf = self.out_conv2(x)
        out_tanh = self.tanh(outputsdf)
        return out_tanh, output
class Decoder_URPC(nn.Module):
    def __init__(self, params):
        super(Decoder_URPC, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)
        self.feature_noise = FeatureNoise()

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        if self.training:
            dp3_out_seg = self.out_conv_dp3(Dropout(x, p=0.5))
        else:
            dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        if self.training:
            dp2_out_seg = self.out_conv_dp2(FeatureDropout(x))
        else:
            dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        if self.training:
            dp1_out_seg = self.out_conv_dp1(self.feature_noise(x))
        else:
            dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg

def masked_average_pooling(feature, mask):
    #print(feature.shape[-2:])
    mask = F.interpolate(mask, size=feature.shape[-2:], mode='bilinear', align_corners=True)
    #print((feature*mask).shape)
    masked_feature = torch.sum(feature * mask, dim=(2, 3)) \
                     / (mask.sum(dim=(2, 3)) + 1e-5)
    return masked_feature

def batch_prototype(feature,mask):  #return B*C*feature_size
    batch_pro = torch.zeros(mask.shape[0], mask.shape[1], feature.shape[1])
    for i in range(mask.shape[1]):
        classmask = mask[:,i,:,:]
        proclass = masked_average_pooling(feature,classmask.unsqueeze(1))
        batch_pro[:,i,:] = proclass
    return batch_pro
def entropy_value(p, C):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=2) / \
        torch.tensor(np.log(C))#.cuda()
    return y1
def agreementmap(similarity_map):
    score_map = torch.argmax(similarity_map,dim=3)
    #score_map =score_map.transpose(1,2)
    ##print(score_map.shape, 'score',score_map[0,0,:])
    gt_onthot = F.one_hot(score_map,6)
    avg_onehot = torch.sum(gt_onthot,dim=2).float()
    avg_onehot = F.normalize(avg_onehot,1.0,dim=2)
    ##print(gt_onthot[0,0,:,:],avg_onehot[0,0,:])
    weight = 1-entropy_value(avg_onehot,similarity_map.shape[3])
    ##print(weight[0,0])
    #score_map = torch.sum(score_map,dim=2)
    return weight
def similarity_calulation(feature,batchpro): #feature_size = B*C*H*W  batchpro= B*C*dim
    B = feature.size(0)
    feature = feature.view(feature.size(0), feature.size(1), -1)  # [N, C, HW]
    feature = feature.transpose(1, 2)  # [N, HW, C]
    feature = feature.contiguous().view(-1, feature.size(2))
    C = batchpro.size(1)
    batchpro = batchpro.contiguous().view(-1, batchpro.size(2))
    feature = F.normalize(feature, p=2.0, dim=1)
    batchpro = F.normalize(batchpro, p=2.0, dim=1).cuda()
    similarity = torch.mm(feature, batchpro.T)
    similarity = similarity.reshape(-1, B, C)
    similarity = similarity.reshape(B, -1, B, C)
    return similarity
def selfsimilaritygen(similarity):
    B = similarity.shape[0]
    mapsize = similarity.shape[1]
    C = similarity.shape[3]
    selfsimilarity = torch.zeros(B,mapsize,C)
    for i in range(similarity.shape[2]):
        selfsimilarity[i,:,:] = similarity[i,:,i,:]
    return selfsimilarity.cuda()
def othersimilaritygen(similarity):
    similarity = torch.exp(similarity)
    for i in range(similarity.shape[2]):
        similarity[i,:,i,:] =0
    similaritysum = torch.sum(similarity,dim=2)
    similaritysum_union = torch.sum(similaritysum,dim=2).unsqueeze(-1)
    #print(similaritysum_union.shape)
    othersimilarity = similaritysum/similaritysum_union
    #print(othersimilarity[1,1,:].sum())
    return othersimilarity
def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x
class Decoder_pro(nn.Module):
    def __init__(self, params):
        super(Decoder_pro, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        print(x.shape,'feature_shape')
        output = self.out_conv(x)
        mask = torch.softmax(output,dim=1)
        batch_pro = batch_prototype(x,mask)
        similarity_map = similarity_calulation(x,batch_pro)
        entropy_weight = agreementmap(similarity_map)
        self_simi_map = selfsimilaritygen(similarity_map) #B*HW*C
        other_simi_map = othersimilaritygen(similarity_map)#B*HW*C
        return output, self_simi_map, other_simi_map, entropy_weight
class DS1(nn.Module):
    def __init__(self, n_prototypes, n_feature_maps):
        super(DS1, self).__init__()
        self.n_prototypes = n_prototypes
        self.w = torch.nn.Linear(in_features=n_feature_maps, out_features=n_prototypes, bias=False).weight
    def forward(self, inputs):
        inputs = inputs.transpose(1, 2)
        inputs = inputs.transpose(2, 3)
        for i in range(self.n_prototypes):
            if i == 0:
                un_mass_i = (self.w[i, :] - inputs) ** 2
                un_mass_i = torch.sum(un_mass_i, dim=-1, keepdim=True)
                print(un_mass_i.shape)
                un_mass = un_mass_i
                #un_mass = torch.unsqueeze(un_mass_i, -1)
            if i >= 1:
                un_mass_i = (self.w[i, :] - inputs) ** 2
                un_mass_i = torch.sum(un_mass_i, dim=-1, keepdim=True)
                un_mass = torch.cat([un_mass, un_mass_i], -1)
        return un_mass
class DistanceActivation_layer(torch.nn.Module):
    '''
    verified
    '''
    def __init__(self, n_prototypes,init_alpha=0,init_gamma=0.1):
        super(DistanceActivation_layer, self).__init__()
        self.eta = torch.nn.Linear(in_features=n_prototypes, out_features=1, bias=False)#.weight.data.fill_(torch.from_numpy(np.array(init_gamma)).to(device))
        self.xi = torch.nn.Linear(in_features=n_prototypes, out_features=1, bias=False)#.weight.data.fill_(torch.from_numpy(np.array(init_alpha)).to(device))
        #torch.nn.init.kaiming_uniform_(self.eta.weight)
        #torch.nn.init.kaiming_uniform_(self.xi.weight)
        torch.nn.init.constant_(self.eta.weight,init_gamma)
        torch.nn.init.constant_(self.xi.weight,init_alpha)
        #self.alpha_test = 1/(torch.exp(-self.xi.weight)+1)
        self.n_prototypes = n_prototypes
        self.alpha = None

    def forward(self, inputs):
        gamma=torch.square(self.eta.weight)
        alpha=torch.neg(self.xi.weight)
        alpha=torch.exp(alpha)+1
        alpha=torch.div(1, alpha)
        self.alpha=alpha
        si=torch.mul(gamma, inputs)
        si=torch.neg(si)
        si=torch.exp(si)
        si = torch.mul(si, alpha)
        max_val, max_idx = torch.max(si, dim=-1, keepdim=True)
        si /= (max_val + 0.0001)
        #print(si.shape,'si')

        return si,alpha
class Belief_layer(torch.nn.Module):
    '''
    verified
    '''
    def __init__(self, n_prototypes, num_class):
        super(Belief_layer, self).__init__()
        self.beta = torch.nn.Linear(in_features=n_prototypes, out_features=num_class, bias=False).weight
        self.num_class = num_class
    def forward(self, inputs):
        beta = torch.square(self.beta)
        beta_sum = torch.sum(beta, dim=0, keepdim=True)
        u = torch.div(beta, beta_sum)
        #print(inputs.shape,u.shape)
        mass_prototype = torch.einsum('cp,b...p->b...pc',u, inputs)
        return mass_prototype
class Omega_layer(torch.nn.Module):
    '''
    verified, give same results
    '''
    def __init__(self, n_prototypes, num_class):
        super(Omega_layer, self).__init__()
        self.n_prototypes = n_prototypes
        self.num_class = num_class

    def forward(self, inputs):
        mass_omega_sum = 1 - torch.sum(inputs, -1, keepdim=True)
        #mass_omega_sum = 1. - mass_omega_sum[..., 0]
        #mass_omega_sum = torch.unsqueeze(mass_omega_sum, -1)
        mass_with_omega = torch.cat([inputs, mass_omega_sum], -1)
        return mass_with_omega
class Dempster_layer(torch.nn.Module):
    '''
    verified give same results
    '''
    def __init__(self, n_prototypes, num_class):
        super(Dempster_layer, self).__init__()
        self.n_prototypes = n_prototypes
        self.num_class = num_class

    def forward(self, inputs):
        m1 = inputs[..., 0, :]
        omega1 = torch.unsqueeze(inputs[..., 0, -1], -1)
        for i in range(self.n_prototypes - 1):
            m2 = inputs[..., (i + 1), :]
            omega2 = torch.unsqueeze(inputs[..., (i + 1), -1], -1)
            combine1 = torch.mul(m1, m2)
            combine2 = torch.mul(m1, omega2)
            combine3 = torch.mul(omega1, m2)
            combine1_2 = combine1 + combine2
            combine2_3 = combine1_2 + combine3
            combine2_3 = combine2_3 / torch.sum(combine2_3, dim=-1, keepdim=True)
            m1 = combine2_3
            omega1 = torch.unsqueeze(combine2_3[..., -1], -1)
        return m1
class DempsterNormalize_layer(torch.nn.Module):
    '''
    verified
    '''
    def __init__(self):
        super(DempsterNormalize_layer, self).__init__()
    def forward(self, inputs):
        mass_combine_normalize = inputs / torch.sum(inputs, dim=-1, keepdim=True)
        return mass_combine_normalize

class DS1_activate(nn.Module):
    def __init__(self,  input_dim):
        super(DS1_activate, self).__init__()
        self.eta = torch.nn.Linear(input_dim, 1, bias=False)#.weight.data.fill_(torch.from_numpy(np.array(init_gamma)).to(device))
        self.xi = torch.nn.Linear(input_dim, 1, bias=False)#.weight.data.fill_(torch.from_numpy(np.array(init_alpha)).to(device))
        #torch.nn.init.kaiming_uniform_(self.eta.weight)
        #torch.nn.init.kaiming_uniform_(self.xi.weight)
        torch.nn.init.constant_(self.eta.weight,0.1)
        torch.nn.init.constant_(self.xi.weight,0)
        #self.xi = nn.Parameter(torch.randn(1, input_dim), requires_grad=True)
        #self.eta = nn.Parameter(torch.randn(1, input_dim), requires_grad=True)
        self.input_dim = input_dim

    def forward(self, inputs):
        gamma = torch.square(self.eta.weight)
        alpha = -self.xi.weight
        alpha = torch.exp(alpha) + 1
        alpha = torch.reciprocal(alpha)
        si = gamma * inputs
        si = -si
        si = torch.exp(si)
        si = si * alpha
        # si = si / (torch.max(si, dim=-1, keepdim=True)[0] + 0.0001)
        return si,alpha



class DS2(nn.Module):
    def __init__(self, input_dim, num_class):
        super(DS2, self).__init__()
        #self.beta = nn.Parameter(torch.randn(input_dim, num_class),requires_grad=True)
        self.beta = torch.nn.Linear(num_class, input_dim, bias=False).weight
        self.input_dim = input_dim
        self.num_class = num_class

    def forward(self, inputs):
        beta = torch.square(self.beta)
        beta_sum = torch.sum(beta, dim=1, keepdim=True)
        u = beta/ beta_sum  ##class probability
        print(u.shape,'uuuu',u.max(dim=1)[1])
        inputs_new = torch.unsqueeze(inputs, -1)
        #print(inputs_new.shape)
        a = inputs_new.expand(-1, -1, -1,  -1,self.num_class-1)
        #print(inputs_new.shape,a.shape)
        inputs_new = torch.cat([a, inputs_new], dim=-1)
        mass_prototype = None
        for i in range(self.input_dim):
            mass_prototype_i = torch.mul(u[i, :], inputs_new[:, :, :, i, :])
            mass_prototype_i = torch.unsqueeze(mass_prototype_i, -2)
            if mass_prototype is None:
                mass_prototype = mass_prototype_i
            else:
                mass_prototype = torch.cat([mass_prototype, mass_prototype_i], dim=-2)
        return mass_prototype
class DS2_omega(nn.Module):
    def __init__(self, input_dim, num_class):
        super(DS2_omega, self).__init__()
        self.input_dim = input_dim
        self.num_class = num_class
    def forward(self, inputs):
        mass_omega_sum = torch.sum(inputs, -1, keepdim=True)
        #print(mass_omega_sum.min(),mass_omega_sum.max())
        mass_omega_sum = 1-mass_omega_sum[:, :, :, :, 0]
        mass_omega_sum = torch.unsqueeze(mass_omega_sum, -1)
        mass_with_omega = torch.cat([inputs, mass_omega_sum], -1)
        return mass_with_omega


class DS3_Dempster(nn.Module):
    def __init__(self, input_dim, num_class):
        super(DS3_Dempster, self).__init__()
        self.input_dim = input_dim
        self.num_class = num_class

    def forward(self, inputs):
        m1 = inputs[:, :, :, 0, :]
        omega1 = torch.unsqueeze(inputs[:, :, :, 0, -1], -1)

        for i in range(self.input_dim - 1):
            m2 = inputs[:, :, :, i + 1, :]
            omega2 = torch.unsqueeze(inputs[:, :, :, i + 1, -1], -1)

            combine1 = torch.mul(m1, m2)
            combine2 = torch.mul(m1, omega2)
            combine3 = torch.mul(omega1, m2)
            combine1_2 = torch.add(combine1, combine2)
            combine2_3 = torch.add(combine1_2, combine3)
            m1 = combine2_3[:, :, :, :-1]
            omega1 = torch.mul(omega1, omega2)
            #omega1 = 1 - torch.sum(m1,dim=-1,keepdim=True)
            print(omega1.max(),'omega',omega1.min())
            m1 = torch.cat([m1, omega1], -1)
        return m1
class DM_pignistic(nn.Module):
    def __init__(self, num_class):
        super(DM_pignistic, self).__init__()
        self.num_class = num_class

    def forward(self, inputs):
        aveage_Pignistic = torch.div(inputs[:, :, :, -1], self.num_class)
        aveage_Pignistic = torch.unsqueeze(aveage_Pignistic, -1)
        mass_class = inputs[:, :, :, 0:-1]
        Pignistic_prob = torch.add(mass_class, aveage_Pignistic)

        return Pignistic_prob,inputs[:, :, :, -1]
class DS3_normalize(nn.Module):
    def __init__(self):
        super(DS3_normalize, self).__init__()

    def forward(self, inputs):
        mass_combine_normalize = inputs / torch.sum(inputs, dim=-1, keepdim=True)
        print(mass_combine_normalize[0,1,1,:])
        return mass_combine_normalize
class Decoder_BF(nn.Module):
    def __init__(self, params):
        super(Decoder_BF, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        self.pro_num = self.params['pro_num']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, mode_upsampling=self.up_type)

        #self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)
        self.DS1 = DS1(self.pro_num, self.ft_chns[0])
        self.DS1_1 = DS1_activate(self.pro_num)
        self.DS2 = DS2(self.pro_num,self.n_class)
        self.DS2_1 = DS2_omega(self.pro_num,self.n_class)
        self.DS3 = DS3_Dempster(self.pro_num,self.n_class)
        self.DS3_1 = DM_pignistic(self.n_class)
        self.DS_out = DS3_normalize()
    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        x = self.DS1(x)
        x, alpha = self.DS1_1(x)
        x = self.DS2(x)
        x = self.DS2_1(x)
        x = self.DS3(x)
        x,uncertainty = self.DS3_1(x)
        x = self.DS_out(x)
        ##print(uncertainty.shape,'feature_shape')
        return x, alpha, uncertainty
class UNet_pro(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_pro, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [32, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params1)
        self.decoder1 = Decoder_pro(params1)

    def forward(self, x):
        feature = self.encoder(x)
        output1,self_simi_map, other_simi_map,entropy_weight = self.decoder1(feature)
        return output1,self_simi_map, other_simi_map,entropy_weight
def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x
class UNet_CCT(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_CCT, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = DecoderCCT(params)
        self.aux_decoder1 = DecoderCCT(params)
        self.aux_decoder2 = DecoderCCT(params)
        self.aux_decoder3 = DecoderCCT(params)

    def forward(self, x):
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)
        aux1_feature = [FeatureNoise()(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)
        aux2_feature = [Dropout(i) for i in feature]
        aux_seg2 = self.aux_decoder2(aux2_feature)
        aux3_feature = [FeatureDropout(i) for i in feature]
        aux_seg3 = self.aux_decoder3(aux3_feature)
        return main_seg, aux_seg1, aux_seg2, aux_seg3


class UNet_URPC(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_URPC, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder_URPC(params)

    def forward(self, x):
        shape = x.shape[2:]
        feature = self.encoder(x)
        dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg = self.decoder(
            feature, shape)
        return dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg
class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)

    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        return output1
class UNet_sdf(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_sdf, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params1)
        self.decoder1 = Decoder_sdf(params1)

    def forward(self, x):
        feature = self.encoder(x)
        outputsdf,output = self.decoder1(feature)
        return outputsdf,output
class BFDCNet2d_v1(nn.Module):
    def __init__(self, in_chns, class_num):
        super(BFDCNet2d_v1, self).__init__()

        params1 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 1,
                   'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 0,
                   'acti_func': 'relu',
                   'pro_num':16}
        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder_BF(params2)

    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        output2,alpha,uncertainty = self.decoder2(feature)
        output2 = output2.transpose(3, 2)
        output2 = output2.transpose(2,1)
        #uncertainty = uncertainty.transpose(3, 2)
        #uncertainty = uncertainty.transpose(2,1)
        print(uncertainty.max(),output2.shape)
        alpha = torch.mean(torch.abs(alpha))
        uncertaintyavg = torch.mean(uncertainty*uncertainty)
        return output1, output2,alpha,uncertainty,uncertaintyavg
class MCNet2d_v1(nn.Module):
    def __init__(self, in_chns, class_num):
        super(MCNet2d_v1, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 0,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)
        
    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        output2 = self.decoder2(feature)
        return output1, output2
    
class MCNet2d_v2(nn.Module):
    def __init__(self, in_chns, class_num):
        super(MCNet2d_v2, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 0,
                  'acti_func': 'relu'}
        params3 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 2,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)
        self.decoder3 = Decoder(params3)
        
    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        output2 = self.decoder2(feature)
        output3 = self.decoder3(feature)
        return output1, output2, output3
class DecoderDiceCE(nn.Module):
    def __init__(self, params):
        super(DecoderDiceCE, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output,x
def DS_Combin_two(alpha1, alpha2,class_num):
        """
        :param alpha1: Dirichlet distribution parameters of view 1
        :param alpha2: Dirichlet distribution parameters of view 2
        :return: Combined Dirichlet distribution parameters
        """
        alpha = dict()
        alpha[0], alpha[1] = alpha1,alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            #print(b[v].shape)
            u[v] = class_num / S[v]#B*C*1

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, class_num, 1), b[1].view(-1, 1, class_num ))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[1].shape) #B*C*1
        #print(uv1_expand.shape,'uv1')
        bu = torch.mul(b[0], uv1_expand)#B*C*1

        # b^1 * u^0
        uv2_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv2_expand)
        # calculate C
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)#B
        #print(bb.shape, 'bb_sum',torch.diagonal(bb, dim1=-2, dim2=-1))
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        C = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1,1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1,1).expand(u[0].shape))

        # calculate new S
        S_a = class_num / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a
class DiceCENet2d_fuse(nn.Module):
    def __init__(self, in_chns, class_num):
        super(DiceCENet2d_fuse, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}
        self.params = params1
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=1, padding=0)
        self.encoder = Encoder(params1)
        self.decoder1 = DecoderDiceCE(params1)
        self.decoder2 = DecoderDiceCE(params1)
        self.decoder3 = DecoderDiceCE(params1)
        self.dropout = nn.Dropout2d(p=0.01, inplace=False)
        self.block_nine= ConvBlock_2d(2,self.ft_chns[0], self.ft_chns[0]) 
        self.BN = nn.ReLU()
    def forward(self, x):
        feature = self.encoder(x)
        out_seg1,x_up1 = self.decoder1(feature)
        out_seg2,x_up2 = self.decoder2(feature)
        out_seg3,x_up3 = self.decoder3(feature)
        evidence1 = F.softplus(out_seg1)
        alpha1 = evidence1+1
        evidence2 = F.softplus(out_seg2)
        alpha2 = evidence2 + 1
        prob2 = alpha2/torch.sum(alpha2,dim=1,keepdim=True)
        resize_alpha1 = alpha1.view(alpha1.size(0), alpha1.size(1), -1)  # [N, C, HW]
        resize_alpha1 = resize_alpha1.transpose(1, 2)  # [N, HW, C]
        resize_alpha1 = resize_alpha1.contiguous().view(-1, resize_alpha1.size(2))
        resize_alpha2 = alpha2.view(alpha2.size(0), alpha2.size(1), -1)  # [N, C, HW]
        resize_alpha2 = resize_alpha2.transpose(1, 2)  # [N, HW, C]
        resize_alpha2 = resize_alpha2.contiguous().view(-1, resize_alpha2.size(2))
        fuse_out_sup = DS_Combin_two( resize_alpha1, resize_alpha2,4)
        fuse_out_sup = fuse_out_sup/torch.sum(fuse_out_sup,dim=1,keepdim=True)
        #fuse_out =self.out_conv(((self.block_nine(((x_up1))+((x_up2)))))) 
        fuse_out = F.softplus(out_seg3)
        fuse_out = fuse_out+1
        #fuse_out = fuse_out.view(fuse_out.size(0), fuse_out.size(1), -1)  # [N, C, HW]
        #fuse_out = fuse_out.transpose(1, 2)  # [N, HW, C]
        #fuse_out = fuse_out.contiguous().view(-1, fuse_out.size(2))
        alpha3_prob = fuse_out/torch.sum(fuse_out,dim=1,keepdim=True)
        #print(fuse_out.max(), fuse_out.min())
        return  alpha1,alpha2,prob2,alpha3_prob,fuse_out_sup,fuse_out
class MCNet2d_v3(nn.Module):
    def __init__(self, in_chns, class_num):
        super(MCNet2d_v3, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 0,
                  'acti_func': 'relu'}
        params3 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 2,
                  'acti_func': 'relu'}
        params4 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 3,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)
        self.decoder3 = Decoder(params3)
        self.decoder4 = Decoder(params4)
        
    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        output2 = self.decoder2(feature)
        output3 = self.decoder3(feature)    
        output4 = self.decoder4(feature)
        return output1, output2, output3, output4

if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from ptflops import get_model_complexity_info
    model = UNet(in_chns=1, class_num=4).cuda()
    with torch.cuda.device(0):
      macs, params = get_model_complexity_info(model, (1, 256, 256), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
      print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
      print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    import ipdb; ipdb.set_trace()
