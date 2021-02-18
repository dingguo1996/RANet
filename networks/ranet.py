import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
affine_par = True
import functools

import sys, os
from utils.pyt_utils import load_model

from ops.match_boundary.modules.match_boundary import MatchBoundary
from ops.match_class.modules.match_class import MatchClass
from ops.follow_cluster.modules.follow_cluster import FollowCluster
from ops.vcount_cluster.modules.vcount_cluster import VcountCluster
from ops.split_repscore.modules.split_repscore import SplitRepscore

from inplace_abn import InPlaceABN, InPlaceABNSync
BatchNorm2d = functools.partial(InPlaceABNSync, activation='identity')
BatchNorm2d_relu = functools.partial(InPlaceABNSync)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual      
        out = self.relu_inplace(out)

        return out

class ASPPModule(nn.Module):
    """
    Reference: 
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """
    def __init__(self, features, inner_features=256, out_features=512, dilations=(12, 24, 36)):
        super(ASPPModule, self).__init__()

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(inner_features))
        self.conv2 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(inner_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                   BatchNorm2d(inner_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                   BatchNorm2d(inner_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                   BatchNorm2d(inner_features))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(out_features),
            nn.Dropout2d(0.1)
            )
        
    def forward(self, x):

        _, _, h, w = x.size()

        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle

class RCB(nn.Module):
    def __init__(self):
        super(RCB, self).__init__()
        self.match_class = MatchClass()
        self.match_boundary = MatchBoundary()
        self.conv_1x1_group = nn.Conv2d(1, 1, 1, bias=True)
        # self.fc_group = nn.Conv2d(1, 1, 1, bias=True)
        nn.init.constant_(self.conv_1x1_group.weight,1)
        nn.init.constant_(self.conv_1x1_group.bias,0)

    def forward(self, semantic_score, boundary_score):
        # compute region attention map
        class_max_prob_A_index = semantic_score.softmax(1).max(1, keepdim=True)[1].int()
        edge_prob = boundary_score.softmax(1)[:, 1:, ...].contiguous()
        semantic_tables = self.match_class(semantic_score.softmax(1), class_max_prob_A_index)  # (bs,9409,9409)
        boundary_tables = self.match_boundary(edge_prob)  # (bs,9409,9409)
        region_attention_tables = (1 - semantic_tables)*(1 - boundary_tables)  # (bs,9409,9409)
        # group process
        region_dicision_tables = (region_attention_tables+region_attention_tables.permute(0,2,1))/2  # (bs,9409,9409)
        return region_attention_tables, region_dicision_tables

class RIB(nn.Module):
    def __init__(self, in_channels=512, out_channels=512,k=8):
        super(RIB, self).__init__()
        self.follow_cluster = FollowCluster(0.8)
        self.vcount_cluster = VcountCluster()
        self.split_repscore = SplitRepscore()
        self.feats_conv = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm2d_relu(512),
            # nn.Dropout2d(0.1),
        )
        self.query_conv = nn.Sequential(nn.Conv2d(512,512, 3, padding=1, bias=False),
                                   BatchNorm2d_relu(512))
        # self.conv_key = nn.Sequential(nn.Conv2d(512,512, 1, bias=False),
        #                                    BatchNorm2d_relu(512))
        self.value_conv = nn.Sequential(nn.Conv2d(512,512, 3, padding=1, bias=False),
                                   BatchNorm2d_relu(512))
        self.final_conv = nn.Sequential(nn.Conv2d(1024,512, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d_relu(512),
                                   nn.Dropout2d(0.1))
        self.collect_conv = nn.Sequential(nn.Conv2d(512,512,3, padding=1,bias=False),BatchNorm2d_relu(512))
        self.interact_conv = nn.Sequential(nn.Conv2d(512,512,3, padding=1,bias=False),BatchNorm2d_relu(512))
        self.distribute_conv = nn.Sequential(nn.Conv2d(512,512,3, padding=1,bias=False),BatchNorm2d_relu(512))
        self.k = k

    def forward(self, feats, region_attention_tables, region_dicision_tables):
        feats = self.feats_conv(feats)
        feats_query = self.query_conv(feats)
        feats_value = self.value_conv(feats)
        region_maps = self.follow_cluster(region_dicision_tables)
        contextual_feats = []
        for bs_idx in range(region_maps.shape[0]):
            region_map = region_maps[bs_idx]
            region_attention_table = region_attention_tables[bs_idx]
            # representative_score = representative_scores[bs_idx]
            feat_query = feats_query[bs_idx]
            feat_value = feats_value[bs_idx]
            # build region image
            with torch.no_grad():
                for i in range(10):
                    region_map = region_map.gather(0, region_map.long())
                cluster_idx = 0
                for cluster_pos in region_map.unique().tolist():
                    region_map = torch.where(region_map == cluster_pos,
                                torch.ones_like(region_map) * (cluster_idx), region_map)
                    cluster_idx +=1

            vcount = self.vcount_cluster(region_attention_table, region_map)  # (num_clusters,9409)
            representative_score = (vcount/((vcount>0).sum(1,keepdim=True).float())).sum(0)
            vtopk_table = vcount.topk(self.k, dim=1).indices.long().reshape(-1).unique()
            reshaped_feat_query = feat_query.reshape((feat_query.shape[0], -1))
            reshaped_feat_value = feat_value.reshape((feat_value.shape[0], -1))

            #intra-region collection
            representative_feat = reshaped_feat_value.permute(1, 0)[vtopk_table]
            collect_w = torch.matmul(representative_feat,reshaped_feat_value*representative_score).softmax(dim=1)
            collect_rep_feat = representative_feat + torch.matmul(collect_w, reshaped_feat_value.permute(1, 0))
            collect_rep_feat = self.collect_conv(collect_rep_feat.reshape([1,*collect_rep_feat.shape,1]).permute(0,2,1,3)).permute(0,2,1,3).reshape(*collect_rep_feat.shape)

            #inter-region interaction
            inter_region_w = torch.matmul(collect_rep_feat, collect_rep_feat.permute(1,0)*representative_score[vtopk_table]).softmax(dim=1)
            interaction_rep_feat = collect_rep_feat + torch.matmul(inter_region_w, collect_rep_feat)
            interaction_rep_feat = self.interact_conv(interaction_rep_feat.reshape([1,*interaction_rep_feat.shape,1]).permute(0,2,1,3)).permute(0,2,1,3).reshape(*interaction_rep_feat.shape)

            #intra-region distribution
            distribute_w = torch.matmul(reshaped_feat_query.permute(1, 0),interaction_rep_feat.permute(1,0)*representative_score[vtopk_table]).softmax(dim=1)
            distribute_feat = reshaped_feat_query + torch.matmul(distribute_w, interaction_rep_feat).permute(1,0)
            contextual_feat = distribute_feat.reshape(feat_query.shape)
            contextual_feat = self.distribute_conv(contextual_feat.unsqueeze(0)).squeeze(0)
            contextual_feats.append(contextual_feat)

        final_feats = self.final_conv(torch.cat((feats,torch.cat(contextual_feats).reshape(feats.shape)),dim=1))
        return final_feats

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, criterion):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(4,8,16))

        # self.head = ASPPModule(2048)
        self.head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=36, dilation=36, bias=False),
            BatchNorm2d(512),
            nn.Dropout2d(0.1),
            )
        self.head_semantic_logit = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.head_boundary_logit = nn.Conv2d(512, 2, kernel_size=1, stride=1, padding=0, bias=True)

        self.rcb = RCB()
        self.rib = RIB()

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(512),
            nn.Dropout2d(0.1),
            )
        self.dsn_semantic_logit = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_boundary_logit = nn.Conv2d(512, 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.criterion = criterion

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion,affine = affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x, labels=None, boundary_labels=None):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_feat = self.dsn(x)
        x_semantic_dsn = self.dsn_semantic_logit(x_feat)
        x_boundary_dsn = self.dsn_boundary_logit(x_feat)
        x = self.layer4(x)
        # x = self.head(x)
        region_attention_tables, region_dicision_tables = self.rcb(x_semantic_dsn, x_boundary_dsn)
        final_feats = self.rib(x, region_attention_tables, region_dicision_tables)
        # x = self.nonLocalValue(x, x_semantic_dsn, x_boundary_dsn)
        x_semantic = self.head_semantic_logit(final_feats)
        x_boundary = self.head_boundary_logit(final_feats)
        outs = [x_semantic, x_boundary, x_semantic_dsn, x_boundary_dsn]

        if self.criterion is not None and labels is not None and boundary_labels is not None:
            return self.criterion(outs, labels, boundary_labels)
        else:
            return outs


def Seg_Model(num_classes, criterion=None, pretrained_model=None):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes, criterion)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)

    return model

