import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from .graph_ntu import Graph

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        kernel_size_1 = kernel_size
        pad_1 = 4
        dilation_1 = 1
        
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            groups=in_channels,
            kernel_size=(kernel_size_1, 1),   # Conv along the temporal dimension only
            padding=(pad_1, 0),
            dilation=(dilation_1,1),
            stride=(stride, 1)
        )     
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 1),   # Conv along the channel dimension only
            stride=(1, 1)
        )   

        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv1)
        conv_init(self.conv2)
        bn_init(self.bn, 1)

    def forward(self, x):
        N, Ce_in, T, V_node = x.shape # (N, Ce_in, T, V_node)
        x = self.conv2(self.conv1(x))
        x = self.bn(x)
        return x
    
class VertexTemporalConv(nn.Module):
    def __init__(self, node_in_channels, node_out_channels, kernel_size=9, stride=1):
        super().__init__()
        self.tempconv = TemporalConv(node_in_channels, node_out_channels, kernel_size, stride)

    def forward(self, fv):
        # `fv` (node features) has shape (N, Cv, T, V_node)
        return self.tempconv(fv)
    
class MGNBlock(nn.Module):
    def __init__(self, node_in_channels, node_out_channels, adj_mat):
        super().__init__()
        
        # Adaptive block with learnable graphs; shapes (V_node, V_node)
        self.adj_mat = nn.Parameter(torch.from_numpy(adj_mat.astype(np.float32))) # default: requires_grad = True
        
        # Updating functions
        self.Hv_agg = nn.Conv2d(2*node_in_channels, node_out_channels, kernel_size = (1,1))
        self.relu = nn.ReLU()
        self.bn_v = nn.BatchNorm2d(node_out_channels)
        bn_init(self.bn_v, 1)
        conv_init(self.Hv_agg)
        
    def forward(self, fv):
        # `fv` (node features) has shape (N, Cv, T, V_node)
        N, Cv_in, T, V_node = fv.shape
        
        # Reshape for matmul, shape: (NT, C, V)
        fv = fv.permute(0,2,1,3).contiguous().view(N*T, Cv_in, V_node) # (NT,Cv_in,V_node)

        # Compute aggregated node features (source and target)
        fv_agg = torch.einsum('ncv,vy->ncy', fv, self.adj_mat).contiguous().view(N, T, Cv_in, V_node).permute(0,2,1,3) # (N,Cv_in,T,V_nodes)
        fv = fv.contiguous().view(N,T, Cv_in, V_node).permute(0,2,1,3) # (N,Cv_in,T,V_nodes)
        fvp = torch.cat((fv,fv_agg), dim = 1) # (N,2Cv_in,T,V_node)
        fvp = self.Hv_agg(fvp)  # (N,Cv_out,T,V_node)
        fvp = self.bn_v(fvp)
        fvp = self.relu(fvp)
        
        return fvp
    
class GraphTemporalConv(nn.Module):
    def __init__(self, node_in_channels, node_out_channels, adj_mat, temp_kernel_size=9, stride=1, residual=True):
        super(GraphTemporalConv, self).__init__()
        self.mgn = MGNBlock(node_in_channels, node_out_channels, adj_mat)
        self.tcn = VertexTemporalConv(node_out_channels, node_out_channels, kernel_size=temp_kernel_size, stride=stride)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda fv: (0)
        elif (node_in_channels == node_out_channels) and (stride == 1):
            self.residual = lambda fv: (fv)
        else:
            self.residual = VertexTemporalConv(node_in_channels, node_out_channels, kernel_size=temp_kernel_size, stride=stride)
            
    def forward(self, fv):
        fv_res = self.residual(fv)
        fv = self.mgn(fv)
        fv = self.tcn(fv)
        fv += fv_res
        return self.relu(fv)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.graph = Graph()
        adj_mat,_ = self.graph.directed_M,self.graph.undirected_M
        
        num_class, vertices, num_person, motion_channels, joint_channels, bone_channels = 60, 25, 2, 3, 3, 3
        self.joint_FC1 = nn.Linear(joint_channels, 64)
        self.joint_FC2 = nn.Linear(64, 32)
        self.motion_FC1 = nn.Linear(motion_channels, 64)
        self.motion_FC2 = nn.Linear(64, 32)
        self.bone_FC1 = nn.Linear(bone_channels, 64)
        self.bone_FC2 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        
        self.data_bn_joint = nn.BatchNorm2d(joint_channels)
        self.data_bn_motion = nn.BatchNorm2d(motion_channels)
        self.data_bn_bone = nn.BatchNorm2d(bone_channels)
        self.data_bn_v = nn.BatchNorm1d(num_person * 32 * vertices)
        
        self.l1 = GraphTemporalConv(32, 32, adj_mat)
        self.l2 = GraphTemporalConv(32, 32, adj_mat)
        self.l3 = GraphTemporalConv(32, 64, adj_mat)
        self.l4 = GraphTemporalConv(64, 64, adj_mat)
        self.l5 = GraphTemporalConv(64, 64, adj_mat)
        self.l6 = GraphTemporalConv(64, 128, adj_mat, stride = 2)
        self.l7 = GraphTemporalConv(128, 128, adj_mat)
        self.l8 = GraphTemporalConv(128, 256, adj_mat, stride = 2)
        
        self.fc = nn.Linear(256, num_class)
        
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.joint_FC1.weight, 0, math.sqrt(2. / 64))
        nn.init.normal_(self.joint_FC2.weight, 0, math.sqrt(2. / 32))
        nn.init.normal_(self.motion_FC1.weight, 0, math.sqrt(2. / 64))
        nn.init.normal_(self.motion_FC2.weight, 0, math.sqrt(2. / 32))
        nn.init.normal_(self.bone_FC1.weight, 0, math.sqrt(2. / 64))
        nn.init.normal_(self.bone_FC2.weight, 0, math.sqrt(2. / 32))
        bn_init(self.data_bn_v, 1)
        bn_init(self.data_bn_joint, 1)
        bn_init(self.data_bn_motion, 1)
        bn_init(self.data_bn_bone, 1)
        
        def count_params(m):
                return sum(p.numel() for p in m.parameters() if p.requires_grad)
        for module in self.modules():
            print('Module:', module)
            print('# Params:', count_params(module))
            print()
        print('Model total number of params:', count_params(self))

    def forward(self, joint_data, joint_motion_data, bone_data, bone_motion_data):
        
        # joint_data has shape (N, Cv, T, V_node, M) where Cv = 3 & V_node = 25
        # joint_motion_data has shape (N, Cv, T, V_node, M) where Cv = 3 & V_node = 25
        # bone_data has shape (N, Cv, T, V_node, M) where Cv = 3 & V_node = 25
        
        N, Cv, T, V_node, M = joint_data.shape # (N, Cv, T, V_node, M)
        joint_data = joint_data.contiguous().view(N, Cv, T*V_node,M) # (N, Cv, T*V_node, M)
        joint_data = self.data_bn_joint(joint_data).contiguous().view(N, Cv, T, V_node,M).permute(0, 4, 2, 3, 1) # (N, M, T, V_node, Cv)
        joint_data = self.relu(self.joint_FC2(self.relu(self.joint_FC1(joint_data)))) # (N, M, T, V_node, Cv)
        joint_motion_data = joint_motion_data.contiguous().view(N, Cv, T*V_node,M) # (N, Cv, T*V_node, M)
        joint_motion_data = self.data_bn_motion(joint_motion_data).contiguous().view(N, Cv, T, V_node,M).permute(0, 4, 2, 3, 1) # (N, M, T, V_node, Cv)
        joint_motion_data = self.relu(self.motion_FC2(self.relu(self.motion_FC1(joint_motion_data)))) # (N, M, T, V_node, Cv)
        bone_data = bone_data.contiguous().view(N, Cv, T*V_node,M) # (N, Cv, T*V_node, M)
        bone_data = self.data_bn_bone(bone_data).contiguous().view(N, Cv, T, V_node,M).permute(0, 4, 2, 3, 1) # (N, M, T, V_node, Cv)
        bone_data = self.relu(self.bone_FC2(self.relu(self.bone_FC1(bone_data)))) # (N, M, T, V_node, Cv)
        fv = (joint_data + joint_motion_data + bone_data).permute(0, 4, 2, 3, 1) # Early fusion of joint features (N, Cv, T, V_node, M)
        
        N, Cv, T, V_node, M = fv.shape # (N, Cv, T, V_node, M)
        fv = fv.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V_node * Cv, T) # (N, M*V_node*Cv,T)
        fv = self.data_bn_v(fv)
        fv = fv.view(N, M, V_node, Cv, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, Cv, T, V_node)
      
        fv = self.l1(fv)
        fv = self.l2(fv)
        fv = self.l3(fv)
        fv = self.l4(fv)
        fv = self.l5(fv)
        fv = self.l6(fv)
        fv = self.l7(fv)
        fv = self.l8(fv)
        
        # Shape: (N*M,C,T,V), C is same for fv/fe
        out_channels = fv.size(1)

        # Performs pooling over both nodes and frames, and over number of persons
        fv = fv.view(N, M, out_channels, -1).mean(3).mean(1)
        fv = self.fc(fv)
        
        return fv