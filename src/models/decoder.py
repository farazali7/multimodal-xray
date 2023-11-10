import torch
import torch.nn as nn

class Upsample(nn.Module):
    def __init__(self, inchn:int,outchn:int,mid=None):
        """
        Upsamples an input and applied convolution operation
        
        Args:
        inchn - Number of input channels
        outchn - Number of output channels
        mid (optional) - Number of middle channels
        """
        
        super(Upsample,self).__init__()
        
        self.inchn = inchn
        self.outchn = outchn
        
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear')
        
        self.conv = nn.Sequential(nn.Conv2d(self.inchn, self.outchn,kernel_size=3,stride=1,padding=1,bias=True),
                                  nn.BatchNorm2d(outchn),
                                  nn.ReLU(inplace=True))
        
        
    def forward(self,x):
        x = self.upsample(x)
        x = self.conv(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self,outchn:int,in_ft=2048,img_size=256,apply_attention=False,emb_dim=0,nheads=0):
        """
        The main Decoder class
        
        Args:
        outchn - The output channel size
        in_ft - Number of input features from the embedding
        img_size = Desired output image size (hxw)
        apply_attention - Should cross attention be performed?
        s1 - Firt sequence 
        s2 - Second sequence of same length
        emb_dim - Embedding dimensions
        nheads - Number of parallel attention heads
        """
        
        super(Decoder,self).__init__()

        self.in_ft = in_ft
        self.outchn = outchn
        self.imsize = img_size
        self.attn = apply_attention
        self.emb = emb_dim
        self.nheads = nheads
        
        self.features,self.ups = [],[]
        while in_ft>=64:
            self.features.append(in_ft)
            in_ft //= 2
        inchn = self.in_ft 

        for channels in self.features[1:]:
            self.ups.append(Upsample(inchn,channels))
            inchn = channels
        self.UpLayers = nn.Sequential(*self.ups)
        
        self.extra = nn.Sequential(nn.Conv2d(inchn, inchn*2 ,kernel_size=3,stride=1,padding=1,bias=True),
                                  nn.BatchNorm2d(inchn*2),
                                  nn.ReLU(inplace=True))
        
        self.cross_attn = torch.nn.MultiheadAttention(self.emb,self.nheads,batch_first=True)
        
        self.mp = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.final = nn.ConvTranspose2d(inchn,self.outchn,kernel_size=1,stride=1,padding=0)
    
    def extra_conv(self,x):
        return self.extra(x)
    
    def Attention(self,s1,s2):
        out,_ = self.cross_attn(key = s1, value = s1, query = s2)
        return out
    
    def forward(self,x=None,s1=None,s2=None):
        """
        The forward method
        
        Args:
        x - Input image (if any)
        s1 - First sequence (image encodings)
        s2 - Second sequence of same length (text embedding)
        
        """
        
        if self.attn:
            x = self.Attention(s1,s2)
        
        # Reshape to a square image
        sz = int(self.emb ** 0.5)
        x = x.view(x.shape[0],x.shape[1],sz,sz)

        #add upsampling
        x = self.UpLayers(x)
        
        #increasing image shape proportionate to output size
        if x.shape[2] != self.imsize:
            scale = self.imsize//x.shape[2]
            extra = nn.Upsample(scale_factor=scale,mode='bilinear')
            x = extra(x)
            
        x = self.final(x)

        return x
    
"""
#test
s1 = torch.randn((1,256,256))
s2 = torch.randn((1,256,256))
decoder = Decoder(1,in_ft=256,apply_attention=True,emb_dim=256,nheads=4,img_size=512)
out = decoder(s1=s1,s2=s2)
print(out.size())

"""

