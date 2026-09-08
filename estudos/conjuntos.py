"""Modelos de conjuntos com máscaras explícitas, usados no Tutorial 06."""
from .comum import *


def ellipse_clouds(n, rng, max_n=22):
    X=np.zeros((n,max_n,3),dtype=np.float32);diam=[];means=[];areas=[];pairs=[]
    for s in range(n):
        m=int(rng.integers(10,max_n+1));a,b=rng.uniform(.4,1,2)
        t=np.sort(rng.uniform(0,2*np.pi,m));phi=rng.uniform(0,np.pi)
        R=np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
        P=np.column_stack((a*np.cos(t),b*np.sin(t)))@R.T+.03*rng.normal(size=(m,2))
        D=distance.squareform(distance.pdist(P));ij=np.unravel_index(np.argmax(D),D.shape)
        X[s,:m,:2]=P;X[s,:m,2]=1
        diam.append(D.max());means.append(D[np.triu_indices(m,1)].mean());pairs.append(ij)
        areas.append(.5*np.sum(P[:,0]*np.roll(P[:,1],-1)-np.roll(P[:,0],-1)*P[:,1]))
    return X,np.array(diam),np.array(means),np.array(areas),np.array(pairs)


def reorder_clouds(X,rng=None,angular=False,reverse=False):
    Z=X.copy()
    for s in range(len(X)):
        n=int(X[s,:,2].sum());P=X[s,:n]
        if angular:
            C=P[:,:2]-P[:,:2].mean(0);ix=np.argsort(np.arctan2(C[:,1],C[:,0]))
        elif reverse:ix=np.arange(n)[::-1]
        else:ix=rng.permutation(n)
        Z[s,:n]=P[ix]
    return Z

class DeepSets(nn.Module):
    def __init__(self,width=32):
        super().__init__();self.embed=nn.Sequential(nn.Linear(2,width),nn.GELU(),nn.Linear(width,width),nn.GELU())
        self.head=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))
    def forward(self,x):
        # Dividir a SOMA por uma constante não a converte em média por conjunto.
        h=(self.embed(x[:,:,:2])*x[:,:,2,None]).sum(1)/22
        return self.head(h)

class FlatSetMLP(nn.Module):
    def __init__(self,width=64,max_n=22):
        super().__init__();self.net=nn.Sequential(nn.Flatten(),nn.Linear(3*max_n,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))
    def forward(self,x):return self.net(x)

class AttentionBlock(nn.Module):
    def __init__(self,width):
        super().__init__();self.norm=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,3*width);self.out=nn.Linear(width,width)
        self.norm2=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))
    def forward(self,h,mask,remove=None):
        q,k,v=self.qkv(self.norm(h)).chunk(3,dim=-1)
        S=q@k.transpose(-1,-2)/np.sqrt(q.shape[-1]);S=S.masked_fill(~mask[:,None,:],-1e9)
        A=S.softmax(-1)
        if remove is not None:
            # Zerar colunas e renormalizar preserva uma distribuição sobre chaves restantes.
            A=A*(~remove[:,None,:]);A=A/A.sum(-1,keepdim=True).clamp_min(1e-12)
        h=h+self.out(A@v);h=h+self.ff(self.norm2(h))
        return h,A

class SetAttention(nn.Module):
    def __init__(self,width=32,layers=1,positional=False,max_n=22):
        super().__init__();self.positional=positional
        self.embed=nn.Sequential(nn.Linear(2,width),nn.GELU(),nn.Linear(width,width))
        self.blocks=nn.ModuleList([AttentionBlock(width) for _ in range(layers)])
        self.norm=nn.LayerNorm(width);self.head=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))
        pos=torch.arange(max_n).float()[:,None];freq=torch.exp(torch.arange(0,width,2).float()*(-np.log(10000.)/width))
        pe=torch.zeros(max_n,width);pe[:,0::2]=torch.sin(pos*freq);pe[:,1::2]=torch.cos(pos*freq)
        self.register_buffer('pe',pe)
    def forward(self,x,return_attention=False,remove=None):
        mask=x[:,:,2]>0;h=self.embed(x[:,:,:2]);As=[]
        if self.positional:h=h+self.pe[None,:x.shape[1]]
        for block in self.blocks:
            h,A=block(h,mask,remove=remove);As.append(A)
        h=self.norm(h);pooled=(h*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True)
        y=self.head(pooled)
        return (y,As) if return_attention else y
