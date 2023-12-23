# +
import torch
import numpy as np
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .weight_init import keras_init
from torchscale.architecture.config import RetNetConfig
from torchscale.architecture.retnet import RetNetDecoder


def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size,dilation=1, activation='relu',bn_momentum=0.01):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        paddingAmount = int(dilation*(kernel_size-1)/2)
        self.convlayer1 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size= kernel_size,dilation=dilation,stride=1,padding=paddingAmount,padding_mode='zeros')
        self.convlayer2 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size= kernel_size,dilation=dilation,stride=1,padding=paddingAmount,padding_mode='zeros')
        self.activate = activation_func(activation)
        self.bn1 = nn.BatchNorm1d(self.in_channels, momentum=bn_momentum)
        self.bn2 = nn.BatchNorm1d(self.in_channels, momentum=bn_momentum)
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        residual = self.shortcut(x)
        x = self.activate(self.bn1(x))
        x = self.convlayer1(x)
        x = self.activate(self.bn2(x))
        x = self.convlayer2(x)
        x += residual
        return x

class ResComboBlock(nn.Module):
    def __init__(self, in_channels, out_channels,res_W,res_dilation,bn_momentum=0.01):
        super().__init__()
        self.comboBlock = nn.Sequential(
        ResidualBlock(in_channels,out_channels,res_W,res_dilation,bn_momentum=bn_momentum),
        ResidualBlock(in_channels,out_channels,res_W,res_dilation,bn_momentum=bn_momentum),
        ResidualBlock(in_channels,out_channels,res_W,res_dilation,bn_momentum=bn_momentum),
        ResidualBlock(in_channels,out_channels,res_W,res_dilation,bn_momentum=bn_momentum)
        )
    
    def forward(self, x):
        return self.comboBlock(x)

class SpliceAI(nn.Module):
    def __init__(self,CL_max,bn_momentum=0.01, **kwargs):
        super().__init__()
        self.n_channels = 32
        self.CL_max = CL_max
        self.res_W = [11,11,21,41]
        res_dilation = [1,4,10,25]
        self.kernel_size = 1
        self.conv_layer_1 = nn.Conv1d(in_channels=4, out_channels=self.n_channels, kernel_size= self.kernel_size,stride=1)
        self.skip_layers = nn.ModuleList([nn.Conv1d(in_channels=self.n_channels, out_channels=self.n_channels, kernel_size= self.kernel_size,stride=1) for i in range(5)])
        self.res_layers = nn.ModuleList([ResComboBlock(in_channels=self.n_channels, out_channels=self.n_channels, res_W=self.res_W[i], res_dilation=res_dilation[i],bn_momentum=bn_momentum) for i in range(4)])
        
        
    def forward(self, features):
        x = self.conv_layer_1(features)
        skip = self.skip_layers[0](x)

        for i,residualUnit in enumerate(self.res_layers):
            x = residualUnit(x)
            skip += self.skip_layers[i+1](x)
            #skip = torch.cat([skip,self.skip_layers[i+1](x)],axis=1)
        
        x_skip = skip[:,:,:]
        #x_cropped = skip[:,:,self.CL_max//2:-self.CL_max//2]
        return x_skip


class SpliceRetNet(nn.Module):
    def __init__(self, CL_max,n_channels=32,maxSeqLength=4*128,depth=6,heads=4,dim_head=32,mlp_dim=512,dropout=0.01,returnFmap=False,bn_momentum=0.01,determenistic=False,crop=True, **kwargs):
        super().__init__()
        self.n_channels = n_channels
        self.CL_max = CL_max
        self.kernel_size = 1
        self.crop = crop
        self.returnFmap = returnFmap
        self.SpliceAI = SpliceAI(CL_max,bn_momentum=bn_momentum).apply(keras_init)
        self.conv_final = nn.Conv1d(in_channels=self.n_channels, out_channels=3, kernel_size= self.kernel_size,stride=1)
        config = RetNetConfig(decoder_layers=depth,
                        decoder_embed_dim=n_channels,
                        decoder_retention_heads=heads,
                        decoder_ffn_embed_dim=mlp_dim,
                        embed_tokens=True)

        #self.RetNet = RetNetModel(config)
        self.RetNet = RetNetDecoder(config)
    
    def forward(self, features):
        ##calculate the tokens with argmax. Do not drop the 'N'
        tokens = torch.argmax(features,dim=1)

        x = self.SpliceAI(features[:,:4,:])
        x = torch.swapaxes(x, 1, 2)
        x = self.RetNet(tokens,token_embeddings=x)
        #x = self.RetNet(inputs_embeds=x, forward_impl='chunkwise', use_cache=True, recurrent_chunk_size=4)
        #x = model(inputs_embeds=x, forward_impl='parallel', use_cache=True)
        #x = x.last_hidden_state
        x = torch.swapaxes(x, 1, 2)
        #m_1 = nn.Softmax(dim=1)
        out = self.conv_final(x)[:,:,(self.CL_max//2):-(self.CL_max//2)]
        return out
        #x = state
        #skip = self.skip_layers[0](x)
                                                
        #for i,transformer in enumerate(self.transformerBlocks): 
        #    x = transformer(x,actions)
        #    skip += self.skip_layers[i+1](x)
        
        #out = m_1(self.conv_final(skip))
        #if self.crop:
        #    out = out[:,:,(self.CL_max//2):-(self.CL_max//2)]
        #if self.returnFmap:
        #    return out,acceptor_actions,donor_actions,acceptor_log_probs,donor_log_probs,skip[:,:,(self.CL_max//2):-(self.CL_max//2)]
        #else:
        #    return out,acceptor_actions,donor_actions,acceptor_log_probs,donor_log_probs


class SpliceAI_10K(nn.Module):
    def __init__(self,CL_max, **kwargs):
        super().__init__()
        n_channels = 32
        self.CL_max = CL_max
        self.res_W = [11,11,21,41]
        res_dilation = [1,4,10,25]
        self.kernel_size = 1
        #self.res_kernel_size = 11
        self.conv_layer_1 = nn.Conv1d(in_channels=4, out_channels=n_channels, kernel_size= self.kernel_size,stride=1)
        self.skip_layers = nn.ModuleList([nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size= self.kernel_size,stride=1) for i in range(5)])
        self.res_layers = nn.ModuleList([ResComboBlock(in_channels=n_channels, out_channels=n_channels, res_W=self.res_W[i], res_dilation=res_dilation[i]) for i in range(4)])
        self.conv_final = nn.Conv1d(in_channels=n_channels, out_channels=3, kernel_size= self.kernel_size,stride=1)
        
    def forward(self, features):
        x = self.conv_layer_1(features)
        skip = self.skip_layers[0](x)

        for i,residualUnit in enumerate(self.res_layers):
            x = residualUnit(x)
            skip += self.skip_layers[i+1](x)
            #skip = torch.cat([skip,self.skip_layers[i+1](x)],axis=1)

        x = skip[:,:,self.CL_max//2:-self.CL_max//2]
        x = self.conv_final(x)
        m = nn.Softmax(dim=1)
        return m(x)

class SpliceAI_small(nn.Module):
    def __init__(self,CL_max, **kwargs):
        super().__init__()
        n_channels = 32
        self.CL_max = CL_max
        self.res_W = [11]
        res_dilation = [1]
        self.kernel_size = 1
        #self.res_kernel_size = 11
        self.conv_layer_1 = nn.Conv1d(in_channels=4, out_channels=n_channels, kernel_size= self.kernel_size,stride=1)
        self.skip_layers = nn.ModuleList([nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size= self.kernel_size,stride=1) for i in range(2)])
        self.res_layers = nn.ModuleList([ResComboBlock(in_channels=n_channels, out_channels=n_channels, res_W=self.res_W[i], res_dilation=res_dilation[i]) for i in range(1)])
        self.conv_final = nn.Conv1d(in_channels=n_channels, out_channels=3, kernel_size= self.kernel_size,stride=1)
        
    def forward(self, features):
        x = self.conv_layer_1(features)
        skip = self.skip_layers[0](x)

        for i,residualUnit in enumerate(self.res_layers):
            x = residualUnit(x)
            skip += self.skip_layers[i+1](x)
            #skip = torch.cat([skip,self.skip_layers[i+1](x)],axis=1)
        

        #x = skip[:,:,self.CL_max//2:-self.CL_max//2]
        #x = self.conv_final(x)
        #m = nn.Softmax(dim=1)
        return skip


class ResNet_40K(nn.Module):
    def __init__(self,CL_max,exonInclusion=False, **kwargs):
        super().__init__()
        n_channels = 32
        self.CL_max = CL_max
        self.res_W = [11,11,21,41,51]
        res_dilation = [1,4,10,25,75]
        self.kernel_size = 1
        self.exonInclusion = exonInclusion
        #self.res_kernel_size = 11
        self.conv_layer_1 = nn.Conv1d(in_channels=4, out_channels=n_channels, kernel_size= self.kernel_size,stride=1)
        self.skip_layers = nn.ModuleList([nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size= self.kernel_size,stride=1) for i in range(6)])
        self.res_layers = nn.ModuleList([ResComboBlock(in_channels=n_channels, out_channels=n_channels, res_W=self.res_W[i], res_dilation=res_dilation[i]) for i in range(5)])
        self.conv_final = nn.Conv1d(in_channels=n_channels, out_channels=3, kernel_size= self.kernel_size,stride=1)
        if exonInclusion:
            self.conv_exon = nn.Conv1d(in_channels=n_channels, out_channels=1, kernel_size= self.kernel_size,stride=1)
        
        
    def forward(self, features):
        x = self.conv_layer_1(features)
        skip = self.skip_layers[0](x)

        for i,residualUnit in enumerate(self.res_layers):
            x = residualUnit(x)
            #if i ==2:
            #    x_2 = x
            skip += self.skip_layers[i+1](x)
            #skip = torch.cat([skip,self.skip_layers[i+1](x)],axis=1)

        x = skip[:,:,self.CL_max//2:-self.CL_max//2]
        m = nn.Softmax(dim=1)
        out = m(self.conv_final(x))
        if self.exonInclusion:
            exon = nn.Sigmoid()(self.conv_exon(x))
            return out, exon
        else:
            return out
