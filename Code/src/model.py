# +
import torch
import numpy as np
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .weight_init import keras_init

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.gate = nn.Linear(dim, inner_dim)
        #self.pos_bias_for_optimizer1 = nn.Parameter(torch.rand(heads))
        #self.pos_bias_for_optimizer2 = nn.Parameter(torch.rand(heads))
        #self.direction_bias_for_optimizer1 = nn.Parameter(torch.rand(heads))
        #self.direction_bias_for_optimizer2 = nn.Parameter(torch.rand(heads))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = (torch.matmul(q, k.transpose(-1, -2)))* self.scale
        attn = self.attend(dots)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = torch.sigmoid(self.gate(x))*out
        return self.to_out(out)


# -

class Policy(nn.Module):
    def __init__(self,n_channels):
        super(Policy, self).__init__()
        self.n_channels = n_channels
        #self.conv_hidden = nn.Conv1d(in_channels=n_channels, out_channels=8, kernel_size= 1,stride=1)
        self.affine1 = nn.Linear(n_channels, 4)
        #self.dropout = nn.Dropout(p=0.1)
        self.affine2 = nn.Linear(4, 2)
        #self.conv_action = nn.Conv1d(in_channels=8, out_channels=1, kernel_size= 1,stride=1)
        #self.saved_log_probs = []
        #self.rewards = []

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.affine1(x)/np.sqrt(self.n_channels)
        x = nn.LeakyReLU()(x)
        action_scores = self.affine2(x)
        return action_scores
        #return torch.nn.Softmax(dim=2)(action_scores)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        self.layerNormLayers = nn.ModuleList([])
        #self.affine = nn.Linear(dim,dim)
        #self.dropout1 = nn.Dropout(dropout)
        #self.dropout2 = nn.Dropout(dropout)
        self.pos_emb = FixedPositionalEmbedding(dim)
        #self.blocksize = 4
        #self.skip_layers = nn.ModuleList([nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size= 1, stride=1) for i in range(depth+1)])

        for _ in range(depth):
            self.layerNormLayers.append(nn.ModuleList([nn.LayerNorm(dim),nn.LayerNorm(dim)]))
            self.layers.append(nn.ModuleList([Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                 FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, state,actions):
        x_in = torch.transpose(state, 1, 2)
        #x_in = self.affine(x_in)
        pos_emb = self.pos_emb(x_in).type_as(x_in)
        x_in_subset = torch.gather(x_in+pos_emb,1,actions)
        x = x_in_subset
        #x_subset,splice_idx = self.top_k_selection(x,conv_final,m_1,k)
        #dist_matrix = torch.cdist(actions[:,:,:1].type(torch.float), actions[:,:,:1].type(torch.float), p=1)
        #dist_matrix = torch.clip(dist_matrix, min=0, max=20000)/20000
        #direction = (actions[:,:,:1].unsqueeze(1).type(torch.float)-actions[:,:,:1].unsqueeze(2).type(torch.float))[:,:,:,0]
        #pos_direction = (direction>0).type(torch.float)
        #neg_direction = (direction<0).type(torch.float)
        #del direction
        for d,(attn, ff) in enumerate(self.layers):
            #a = self.dropout1(attn(self.layerNormLayers[d][0](x),dist_matrix))
            #b = self.dropout2(ff(self.layerNormLayers[d][1](a+x)))
            x = attn(self.layerNormLayers[d][0](x)) + x
            x = ff(self.layerNormLayers[d][1](x)) + x
            #x = a+b+x
            #if d%self.blocksize==self.blocksize-1:
            #    x = x+x_in
        x = self.expand_sub_tensor(x-x_in_subset,x_in,actions)
        return torch.transpose(x, 1, 2)
    
    def top_k_selection(self,x,conv_final,m_1,k):
        #if self.sampleFromGumbel:
        attention = m_1(conv_final(torch.transpose(x, 1, 2)))
        acceptors = attention[:,1,:]
        donors = attention[:,2,:]
        
        #topk, splice_idx = torch.topk(acceptors+donors, k, dim=1, largest=True, sorted=False)
        #splice_idx = splice_idx.unsqueeze(2).repeat(1,1,self.dim)
        acceptors_sorted = torch.argsort(acceptors,dim=1,descending=True)[:,:k//2:]
        donors_sorted = torch.argsort(donors,dim=1,descending=True)[:,:k//2]
        splice_idx = torch.cat([acceptors_sorted,donors_sorted],dim=1)
        splice_idx = torch.sort(splice_idx,dim=1).values
        splice_idx = splice_idx.unsqueeze(2).repeat(1,1,self.dim)
        return torch.gather(x,1,splice_idx),splice_idx

    def expand_sub_tensor(self,x_subset,x,splice_idx):
        tmp = torch.zeros_like(x)
        return tmp.scatter_(1, splice_idx, x_subset)+x
        #return torch.transpose(x_subset, 1, 2)

class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]

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


class SpliceFormer(nn.Module):
    def __init__(self, CL_max,n_channels=32,maxSeqLength=4*128,depth=4,n_transformer_blocks=2,heads=4,dim_head=32,mlp_dim=512,dropout=0.01,returnFmap=False,bn_momentum=0.01,determenistic=False,crop=True, **kwargs):
        super().__init__()
        self.n_channels = n_channels
        self.CL_max = CL_max
        self.kernel_size = 1
        self.crop = crop
        self.returnFmap = returnFmap
        self.SpliceAI = SpliceAI(CL_max,bn_momentum=bn_momentum).apply(keras_init)
        self.conv_final = nn.Conv1d(in_channels=self.n_channels, out_channels=3, kernel_size= self.kernel_size,stride=1)
        #self.n_transformer_blocks = n_transformer_blocks
        self.maxSeqLength = maxSeqLength
        self.policy = Policy(n_channels=n_channels)
        self.determenistic = determenistic
        self.skip_layers = nn.ModuleList([nn.Conv1d(in_channels=self.n_channels, out_channels=self.n_channels, kernel_size= self.kernel_size,stride=1) for i in range(n_transformer_blocks+1)])
        self.transformerBlocks =  nn.ModuleList([Transformer(self.n_channels, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout) for i in range(n_transformer_blocks)])
        #self.transformerBlocks = nn.ModuleList([SpliceFormerBlock(n_channels=n_channels,maxSeqLength=maxSeqLength,depth=depth,heads=heads,dim_head=dim_head,mlp_dim=mlp_dim,dropout=dropout,sampleFromGumbel=sampleFromGumbel,gumbel_scale=gumbel_scale) for i in range(n_transformer_blocks)])
        
    
    def forward(self, features):
        state = self.SpliceAI(features)
        #print(state.shape)
        m_1 = nn.Softmax(dim=1)
        #out1 = m_1(self.conv_final(x))
        
        
        actions,acceptor_actions,donor_actions,acceptor_log_probs,donor_log_probs = self.select_action(state)
        
        x = state
        skip = self.skip_layers[0](x)
                                                
        for i,transformer in enumerate(self.transformerBlocks): 
            x = transformer(x,actions)
            skip += self.skip_layers[i+1](x)
        
        out = m_1(self.conv_final(skip))
        if self.crop:
            out = out[:,:,(self.CL_max//2):-(self.CL_max//2)]
        if self.returnFmap:
            return out,acceptor_actions,donor_actions,acceptor_log_probs,donor_log_probs,skip[:,:,(self.CL_max//2):-(self.CL_max//2)]
        else:
            return out,acceptor_actions,donor_actions,acceptor_log_probs,donor_log_probs
        
    def select_action(self,state):
        c = torch.zeros_like(state[:,0,:]).detach()
        policy_logits = self.policy(state)
        actions = []
        acceptor_actions = []
        donor_actions = []
        acceptor_log_probs = []
        donor_log_probs = []
        if self.determenistic:
            #log_probs, actions = torch.topk(policy_logits, self.maxSeqLength, dim=1, largest=True, sorted=False)
            #return actions.unsqueeze(2).repeat(1,1,self.n_channels), log_probs
            #for i in range(self.maxSeqLength//2):
            acceptor_logits = policy_logits[:,:,0]
            donor_logits = policy_logits[:,:,1]
            acceptor_order = torch.argsort(torch.argsort(acceptor_logits,dim=1),dim=1)
            donor_order = torch.argsort(torch.argsort(donor_logits,dim=1),dim=1)
            acceptor_logits[acceptor_order-donor_order<0] = -float('inf')
            donor_logits[donor_order-acceptor_order<=0] = -float('inf')
            acceptor_log_probs, acceptor_actions = torch.topk(acceptor_logits, self.maxSeqLength//2, dim=1, largest=True, sorted=False)
            donor_log_probs, donor_actions = torch.topk(donor_logits, self.maxSeqLength//2, dim=1, largest=True, sorted=False)
            actions = torch.cat([acceptor_actions,donor_actions],dim=1)
            return actions.unsqueeze(2).repeat(1,1,self.n_channels),acceptor_actions,donor_actions,acceptor_log_probs,donor_log_probs
            #acceptor_log_probs, acceptor_actions = torch.topk(policy_logits[:,:,0], self.maxSeqLength//2, dim=1, largest=True, sorted=False)
            #donor_log_probs, donor_actions = torch.topk(policy_logits[:,:,1], self.maxSeqLength//2, dim=1, largest=True, sorted=False)
            #log_prob,action = torch.max(policy_logits[:,:,0]-1e5*c,dim=1)
            #acceptor_log_probs.append(log_prob.unsqueeze(1))
            #actions.append(action.unsqueeze(1))
            #acceptor_actions.append(action.unsqueeze(1))
            #c[torch.arange(c.size(0), dtype=torch.long),action] = 1
            #log_prob,action = torch.max(policy_logits[:,:,1]-1e5*c,dim=1)
            #donor_log_probs.append(log_prob.unsqueeze(1))
            #actions.append(action.unsqueeze(1))
            #donor_actions.append(action.unsqueeze(1))
            #c[torch.arange(c.size(0), dtype=torch.long),action] = 1
            #actions = torch.cat([acceptor_actions,donor_actions],dim=1)
            #return actions.unsqueeze(2).repeat(1,1,self.n_channels),acceptor_actions,donor_actions,acceptor_log_probs,donor_log_probs
        else:
            for i in range(self.maxSeqLength//2):
                m = torch.distributions.Categorical(logits=policy_logits[:,:,0]-torch.nan_to_num(float('inf')*c, nan=0))
                action = m.sample()
                acceptor_log_probs.append(m.log_prob(action).unsqueeze(1))
                actions.append(action.unsqueeze(1))
                acceptor_actions.append(action.unsqueeze(1))
                c[torch.arange(c.size(0), dtype=torch.long),action] = 1
                
                m = torch.distributions.Categorical(logits=policy_logits[:,:,1]-torch.nan_to_num(float('inf')*c, nan=0))
                action = m.sample()
                donor_log_probs.append(m.log_prob(action).unsqueeze(1))
                actions.append(action.unsqueeze(1))
                donor_actions.append(action.unsqueeze(1))
                c[torch.arange(c.size(0), dtype=torch.long),action] = 1
                
            return torch.hstack(actions).unsqueeze(2).repeat(1,1,self.n_channels),torch.hstack(acceptor_actions),torch.hstack(donor_actions),torch.hstack(acceptor_log_probs),torch.hstack(donor_log_probs)


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
