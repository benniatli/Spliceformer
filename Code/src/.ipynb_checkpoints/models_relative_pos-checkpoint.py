import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
from .weight_init import keras_init

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

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

# +
def exists(val):
    return val is not None

# relative positional encoding functions

def get_positional_features_exponential(positions, features, seq_len, min_half_life = 3.):
    max_range = math.log(seq_len) / math.log(2.)
    half_life = 2 ** torch.linspace(min_half_life, max_range, features, device = positions.device)
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.) / half_life * positions)

def get_positional_features_central_mask(positions, features, seq_len):
    center_widths = 2 ** torch.arange(1, features + 1, device = positions.device).float()
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).float()

def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = torch.xlogy(concentration - 1., x) - rate * x
    log_normalization = (torch.lgamma(concentration) - concentration * torch.log(rate))
    return torch.exp(log_unnormalized_prob - log_normalization)

def get_positional_features_gamma(positions, features, seq_len, stddev = None, start_mean = None, eps = 1e-8):
    if not exists(stddev):
        stddev = seq_len / (2 * features)

    if not exists(start_mean):
        start_mean = seq_len / features

    mean = torch.linspace(start_mean, seq_len, features, device = positions.device)
    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev ** 2
    probabilities = gamma_pdf(positions.float().abs()[..., None], concentration, rate)
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities)
    return outputs

def get_positional_embed(seq_len, feature_size, device):
    distances = torch.arange(-seq_len + 1, seq_len, device = device)

    feature_functions = [
        get_positional_features_exponential,
        get_positional_features_central_mask,
        get_positional_features_gamma
    ]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(f'feature size is not divisible by number of components ({num_components})')

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len))

    embeddings = torch.cat(embeddings, dim = -1)
    embeddings = torch.cat((embeddings, torch.sign(distances)[..., None] * embeddings), dim = -1)
    return embeddings

def relative_shift(x):
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim = -1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)
    return x[..., :((t2 + 1) // 2)]


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_rel_pos_features,
        heads = 8,
        dim_key = 64,
        dim_value = 64,
        dropout = 0.,
        pos_dropout = 0.
    ):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim_key * heads, bias = False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias = False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias = False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # relative positional encoding

        self.num_rel_pos_features = num_rel_pos_features

        self.to_rel_k = nn.Linear(num_rel_pos_features, dim_key * heads, bias = False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts

        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, positions):
        n, h, device = x.shape[-2], self.heads, x.device

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        content_logits = einsum('b h i d, b h j d -> b h i j', q + self.rel_content_bias, k)
        #distance = splice_idx.unsqueeze(2)-splice_idx+45000.type(torch.int64)
        
        positions = self.pos_dropout(positions)
        rel_k = self.to_rel_k(positions)
        del positions
        
        rel_k = rearrange(rel_k, 'b n m (h d) -> b h n m d', h = h)
        
        rel_logits = einsum('b h i d, b h i j d -> b h i j', q + self.rel_pos_bias, rel_k)
        #rel_logits = torch.tensordot((q + self.rel_pos_bias).unsqueeze(2), rel_k,dims=([4],[4]))
        #rel_logits = relative_shift(rel_logits)
        #(6,4,512,45000)
        #(6,4,512,512)
        #rel_logits = torch.gather(rel_logits,3,splice_idx.unsqueeze(1).unsqueeze(1))

        logits = content_logits + rel_logits
        #logits = content_logits
        attn = logits.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class AttentionOld(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.gate = nn.Linear(dim, inner_dim)

        self.to_out = nn.Sequential(
            
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        #indices = torch.triu_indices(dots.shape[2], dots.shape[3], offset=1)
        #dots[:,:, indices[0], indices[1]] = float('-inf')
        
        attn = self.attend(dots)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = torch.sigmoid(self.gate(x))*out
        return self.to_out(out)


# +
class Transformer(nn.Module):
    def __init__(self, dim,num_rel_pos_features, depth, heads, dim_head, mlp_dim, dropout = 0., pos_dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_rel_pos_features= num_rel_pos_features, heads = heads, dim_key = dim_head, dim_value = dim_head, dropout = dropout, pos_dropout = pos_dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x,positions):
        for attn, ff in self.layers:
            x = attn(x,positions=positions) + x
            x = ff(x) + x
        return x
    
class OldTransformer(nn.Module):
    def __init__(self, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, AttentionOld(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x,splice_idx):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# -

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
    def __init__(self, in_channels, out_channels,kernel_size,dilation=1, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        paddingAmount = int(dilation*(kernel_size-1)/2)
        self.convlayer1 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size= kernel_size,dilation=dilation,stride=1,padding=paddingAmount,padding_mode='zeros')
        self.convlayer2 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size= kernel_size,dilation=dilation,stride=1,padding=paddingAmount,padding_mode='zeros')
        self.activate = activation_func(activation)
        self.bn1 = nn.BatchNorm1d(self.in_channels, momentum=0.01)
        self.bn2 = nn.BatchNorm1d(self.in_channels, momentum=0.01)
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
    def __init__(self, in_channels, out_channels,res_W,res_dilation):
        super().__init__()
        self.comboBlock = nn.Sequential(
        ResidualBlock(in_channels,out_channels,res_W,res_dilation),
        ResidualBlock(in_channels,out_channels,res_W,res_dilation),
        ResidualBlock(in_channels,out_channels,res_W,res_dilation),
        ResidualBlock(in_channels,out_channels,res_W,res_dilation)
        )
    
    def forward(self, x):
        return self.comboBlock(x)

class smallModel(nn.Module):
    def __init__(self,CL_max, **kwargs):
        super().__init__()
        self.n_channels = 32
        self.CL_max = CL_max
        self.res_W = [11,11,21,41]
        res_dilation = [1,4,10,25]
        self.kernel_size = 1
        self.conv_layer_1 = nn.Conv1d(in_channels=4, out_channels=self.n_channels, kernel_size= self.kernel_size,stride=1)
        self.skip_layers = nn.ModuleList([nn.Conv1d(in_channels=self.n_channels, out_channels=self.n_channels, kernel_size= self.kernel_size,stride=1) for i in range(5)])
        self.res_layers = nn.ModuleList([ResComboBlock(in_channels=self.n_channels, out_channels=self.n_channels, res_W=self.res_W[i], res_dilation=res_dilation[i]) for i in range(4)])
        self.conv_final = nn.Conv1d(in_channels=self.n_channels, out_channels=3, kernel_size= self.kernel_size,stride=1)
        
    def forward(self, features):
        x = self.conv_layer_1(features)
        skip = self.skip_layers[0](x)

        for i,residualUnit in enumerate(self.res_layers):
            x = residualUnit(x)
            #if i ==2:
            #    x_2 = x
            skip += self.skip_layers[i+1](x)
            #skip = torch.cat([skip,self.skip_layers[i+1](x)],axis=1)
        
        x_skip = skip[:,:,:]
        x_cropped = skip[:,:,self.CL_max//2:-self.CL_max//2]
        #x = self.conv_final(x_cropped)
        #m_1 = nn.Softmax(dim=1)
        #out_1 = m_1(x/temp)
        #attention = m_1(self.conv_final(x_skip)/temp)
        return x_cropped, x_skip

class SpliceFormerBlock(nn.Module):
    def __init__(self,n_channels=32,maxSeqLength=4*128,depth=4,heads=4,dim_head=32,mlp_dim=512, **kwargs):
        super().__init__()
        self.n_channels = n_channels
        self.maxSeqLength = maxSeqLength
        self.num_rel_pos_features = 6*5
        #self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=32, dim_feedforward=128*4, nhead=8,norm_first=True), num_layers=12)
        self.transformer =  Transformer(self.n_channels,num_rel_pos_features=self.num_rel_pos_features, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=0.01, pos_dropout = 0.05)
        
    def forward(self, x_skip,attention):
        device = x_skip.device
        x_emb1 = torch.transpose(x_skip, 1, 2)
        x_emb = x_emb1
        #x_emb = x_emb1 +  pos_emb
        acceptors_sorted = torch.argsort(attention[:,1,:],dim=1,descending=True)[:,:self.maxSeqLength//2:]
        donors_sorted = torch.argsort(attention[:,2,:],dim=1,descending=True)[:,:self.maxSeqLength//2]
        splice_idx = torch.cat([acceptors_sorted,donors_sorted],dim=1)
        splice_idx_1 = torch.sort(splice_idx,dim=1).values
        difference = splice_idx_1.unsqueeze(2)-splice_idx_1.unsqueeze(1)+45000-1
        positions = get_positional_embed(45000, self.num_rel_pos_features, device).unsqueeze(0).unsqueeze(-2)
        positions_exp = positions.expand(splice_idx.shape[0],positions.shape[1],difference.shape[-1],positions.shape[3])
        positions = torch.gather(positions_exp,1,difference.unsqueeze(-1).repeat(1,1,1,positions_exp.shape[-1]))
        del positions_exp
        splice_idx = splice_idx_1.unsqueeze(2).repeat(1,1,self.n_channels)
        
        embedding = torch.gather(x_emb,1,splice_idx)
        embedding = self.transformer(embedding,positions)
        tmp = torch.zeros_like(x_emb1)
        embedding = tmp.scatter_(1, splice_idx, embedding) 
        embedding = torch.transpose(embedding, 1, 2)
        #embedding = x_skip+embedding
        return embedding

class SpliceFormer(nn.Module):
    def __init__(self, CL_max,n_channels=32,maxSeqLength=4*128,depth=4,heads=4,dim_head=32,mlp_dim=512, **kwargs):
        super().__init__()
        self.n_channels = n_channels
        self.CL_max = CL_max
        self.res_W = [11,11,21,41]
        res_dilation = [1,4,10,25]
        self.kernel_size = 1
        self.smallModel = smallModel(CL_max).apply(keras_init)
        self.skip_layers = nn.ModuleList([nn.Conv1d(in_channels=self.n_channels, out_channels=self.n_channels, kernel_size= self.kernel_size,stride=1) for i in range(2)])
        self.res_layers = nn.ModuleList([ResComboBlock(in_channels=self.n_channels, out_channels=self.n_channels, res_W=self.res_W[i], res_dilation=res_dilation[i]) for i in range(1)])
        self.conv_final = nn.Conv1d(in_channels=self.n_channels, out_channels=3, kernel_size= self.kernel_size,stride=1)
        #self.pos_emb = FixedPositionalEmbedding(self.n_channels)
        self.transformerBlock1 = SpliceFormerBlock(n_channels=n_channels,maxSeqLength=maxSeqLength,depth=depth,heads=heads,dim_head=dim_head,mlp_dim=mlp_dim)
        self.transformerBlock2 = SpliceFormerBlock(n_channels=n_channels,maxSeqLength=maxSeqLength,depth=depth,heads=heads,dim_head=dim_head,mlp_dim=mlp_dim)
        
        
    
    def forward(self, features):
        out1,x_skip = self.smallModel(features)
        m_1 = nn.Softmax(dim=1)
        out1 = m_1(self.conv_final(out1))
        attention =  m_1(self.conv_final(x_skip))
        #pos_emb = self.pos_emb(torch.transpose(x_skip, 1, 2)).type_as(torch.transpose(x_skip, 1, 2))
        emb1 = self.transformerBlock1(x_skip,attention)
        
        tmp = x_skip+emb1
        attention =  m_1(self.conv_final(tmp)) 
        emb2 = self.transformerBlock2(tmp,attention)
        
        #tmp = x_skip+emb1+emb2
        #attention =  m_1(self.conv_final(tmp))
        #emb3 = self.transformerBlock3(tmp,attention,pos_emb)
        #attention =  m_1(self.conv_final(x_skip))
        
        #tmp = x_skip+emb1+emb2+emb3
        #attention =  m_1(self.conv_final(tmp))
        #emb4 = self.transformerBlock4(tmp,attention,pos_emb)
        
        #tmp = x_skip+emb1+emb2+emb3+emb4
        #attention =  m_1(self.conv_final(tmp))
        #emb5 = self.transformerBlock5(tmp,attention,pos_emb)
        
        x_skip = x_skip+emb1+emb2
        #x_skip = x_skip+emb1+emb2+emb3+emb4
        m_2 = nn.Softmax(dim=1)
        out_2 = m_2(self.conv_final(x_skip))
        out_2 = out_2[:,:,(self.CL_max//2):-(self.CL_max//2)]
        #print(out_2.shape)
        #out_2 = torch.zeros_like(out_1)
        #out_2 = m_2(x_skip+x_emb)
        return out1,out_2


# +
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
    
class ResNet_40K(nn.Module):
    def __init__(self,CL_max, **kwargs):
        super().__init__()
        n_channels = 32
        self.CL_max = CL_max
        self.res_W = [11,11,21,41,51]
        res_dilation = [1,4,10,25,75]
        self.kernel_size = 1
        #self.res_kernel_size = 11
        self.conv_layer_1 = nn.Conv1d(in_channels=4, out_channels=n_channels, kernel_size= self.kernel_size,stride=1)
        self.skip_layers = nn.ModuleList([nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size= self.kernel_size,stride=1) for i in range(6)])
        self.res_layers = nn.ModuleList([ResComboBlock(in_channels=n_channels, out_channels=n_channels, res_W=self.res_W[i], res_dilation=res_dilation[i]) for i in range(5)])
        self.conv_final = nn.Conv1d(in_channels=n_channels, out_channels=3, kernel_size= self.kernel_size,stride=1)
        #self.extra_layer = nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size= self.kernel_size,stride=1)
        #self.bn_extra = nn.BatchNorm1d(n_channels, momentum=0.01)
        
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
        x = self.conv_final(x)
        m = nn.Softmax(dim=1)
        return m(x)
