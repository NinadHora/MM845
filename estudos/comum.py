"""Rotinas compartilhadas das resoluções de MM845.

Material de estudo produzido com assistência de IA. Os experimentos e os testes
são reproduzíveis; os números não são garantias universais sobre os métodos.
"""
from pathlib import Path
import os, sys, json, math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.spatial import ConvexHull, cKDTree, distance
from scipy import linalg, sparse
from scipy.special import sph_harm_y
from itertools import combinations_with_replacement
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from threadpoolctl import threadpool_limits

# Limitar threads evita que problemas pequenos sofram com paralelismo excessivo.
torch.set_num_threads(1)
THREAD_LIMIT = threadpool_limits(limits=1)
SEED = 20260908

def sphere(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    if n < 1 or dim < 2: raise ValueError('n >= 1 e dim >= 2 são necessários.')
    x = rng.normal(size=(n, dim))
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def fibonacci(n: int) -> np.ndarray:
    i = np.arange(n); z = 1 - 2*(i+.5)/n
    t = 2*np.pi*i/((1+np.sqrt(5))/2)
    r = np.sqrt(1-z*z)
    return np.column_stack((r*np.cos(t),r*np.sin(t),z))

def geodesic(X: np.ndarray) -> np.ndarray:
    D = np.arccos(np.clip(X@X.T,-1,1)); np.fill_diagonal(D,0)
    return D

def poly(X: np.ndarray, degree: int, return_degrees=False):
    # Potências pré-calculadas evitam montar repetidamente produtos de 25 fatores.
    if degree < 0: raise ValueError('Grau não negativo esperado.')
    n,d=X.shape; powers = [np.power(X[:,j:j+1],np.arange(degree+1)) for j in range(d)]
    cols=[np.ones(n)]; degs=[0]
    for deg in range(1,degree+1):
        for combo in combinations_with_replacement(range(d),deg):
            exps=np.bincount(combo,minlength=d); v=np.ones(n)
            for j,e in enumerate(exps): v *= powers[j][:,e]
            cols.append(v); degs.append(deg)
    F=np.column_stack(cols)
    return (F,np.array(degs)) if return_degrees else F

def real_harmonics(X: np.ndarray, L: int, return_degrees=False):
    """Harmônicos REAIS, ortonormais para a probabilidade dA/(4*pi) em S².

    SciPy: sph_harm_y(l,m,theta_polar,phi_azimutal). Cada bloco tem 2*l+1
    funções. Não confundir uma QR genérica com a decomposição harmônica.
    """
    theta=np.arccos(np.clip(X[:,2],-1,1)); phi=np.arctan2(X[:,1],X[:,0])
    cols=[]; degrees=[]
    for l in range(L+1):
        cols.append(np.sqrt(4*np.pi)*sph_harm_y(l,0,theta,phi).real);degrees.append(l)
        for m in range(1,l+1):
            Y=sph_harm_y(l,m,theta,phi)*np.sqrt(8*np.pi)
            cols.extend([Y.real,Y.imag]);degrees.extend([l,l])
    F=np.column_stack(cols)
    return (F,np.array(degrees)) if return_degrees else F

def target(X):
    x,y,z=X.T
    return .6*(2*z*z-x*x-y*y)+1.1*x*y+.9*(x**3-3*x*y*y)

BUMP=np.array([.4,.5,.76]); BUMP/=np.linalg.norm(BUMP)
def bump(X,w=.28): return np.exp(-(2-2*(X@BUMP))/(2*w*w))

def table(rows, name: str, out: Path):
    df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/(name+'.csv'),index=False)
    print(df.to_string(index=False,max_rows=30))
    return df

def figure(name: str, out: Path):
    """Salva e mostra a figura atual; uma figura por gráfico, sem paleta imposta."""
    out.mkdir(parents=True,exist_ok=True)
    plt.tight_layout(); plt.savefig(out/(name+'.png'),dpi=140,bbox_inches='tight')
    plt.show(); plt.close()

def write_summary(path: Path, objective,idea,method,result,limit):
    text=(f'# Mini-relatório\n\n**Objetivo:** {objective}\n\n'
          f'**Ideia matemática:** {idea}\n\n**Experimento:** {method}\n\n'
          f'**Resultado:** {result}\n\n**Limite:** {limit}\n\n'
          '[Resoluções e saídas](solucoes.ipynb) · [Código executável](solucoes.py)\n')
    path.write_text(text,encoding='utf-8')

def mlp(d_in,width=24,depth=2,act=nn.Tanh,seed=0):
    torch.manual_seed(seed); layers=[];d=d_in
    for _ in range(depth):layers.extend([nn.Linear(d,width),act()]);d=width
    layers.append(nn.Linear(d,1))
    return nn.Sequential(*layers)

def predict(model, X, batch_size=512):
    model.eval();out=[]
    with torch.no_grad():
        for j in range(0,len(X),batch_size):
            out.append(model(torch.as_tensor(X[j:j+batch_size],dtype=torch.float32)).cpu().numpy().ravel())
    return np.concatenate(out)

def train(model,X,y,epochs=500,lr=.003,batch_size=None,optimizer='adam',shuffle_seed=0,
          loss_kind='mse',augment=None,record_every=1):
    """Treino; nenhuma escolha de hiperparâmetro usa o conjunto de teste."""
    X=torch.as_tensor(X,dtype=torch.float32);y=torch.as_tensor(y,dtype=torch.float32).reshape(-1,1)
    opt=(torch.optim.Adam if optimizer=='adam' else torch.optim.SGD)(model.parameters(),lr=lr)
    gen=torch.Generator().manual_seed(shuffle_seed)
    bs=batch_size or len(X);loss_fn=nn.MSELoss() if loss_kind=='mse' else nn.BCEWithLogitsLoss()
    hist=[];model.train()
    for epoch in range(epochs):
        order=torch.randperm(len(X),generator=gen) if bs<len(X) else torch.arange(len(X))
        total=0.
        for start in range(0,len(X),bs):
            ix=order[start:start+bs];xb=X[ix]
            if augment is not None:xb=augment(xb,gen)
            opt.zero_grad();loss=loss_fn(model(xb),y[ix]);loss.backward();opt.step()
            total += float(loss.detach())*len(ix)
        if epoch%record_every==0:hist.append((epoch+1,total/len(X)))
    return np.asarray(hist)

def curves(n,rng,K=8,sigma=.25,decay=2.5,ntheta=256):
    k=np.arange(1,K+1);th=np.linspace(0,2*np.pi,ntheta,endpoint=False)
    C=np.cos(k[:,None]*th);S=np.sin(k[:,None]*th)
    a=rng.normal(size=(n,K))*sigma/k**decay;b=rng.normal(size=(n,K))*sigma/k**decay
    r=1+a@C+b@S
    # Rejeitar raios não positivos: uma curva polar simples é pressuposto do rótulo.
    bad=np.min(r,axis=1)<=0
    while bad.any():
        a[bad]=rng.normal(size=(bad.sum(),K))*sigma/k**decay
        b[bad]=rng.normal(size=(bad.sum(),K))*sigma/k**decay
        r[bad]=1+a[bad]@C+b[bad]@S;bad=np.min(r,axis=1)<=0
    rp=(-a*k)@S+(b*k)@C;rpp=(-a*k*k)@C+(-b*k*k)@S
    P=a*a+b*b
    return dict(a=a,b=b,P=P,r=r,area=np.pi+.5*np.pi*P.sum(1),
                length=2*np.pi*np.sqrt(r*r+rp*rp).mean(1),
                convex=((r*r+2*rp*rp-r*rpp).min(1)>=0).astype(float),k=k)

def raster_curves(n,rng,grid=32):
    D=curves(n,rng,ntheta=512)
    yy,xx=np.mgrid[:grid,:grid];x=(xx+.5)/grid*3-1.5;y=(yy+.5)/grid*3-1.5
    angle=np.arctan2(y,x)%(2*np.pi);ix=(angle/(2*np.pi)*512).astype(int)
    X=(np.hypot(x,y)[None]<=D['r'][:,ix]).astype(np.float32)[:,None]
    return X,D['area'],D['convex']

def shifts(X,rng,max_shift=8):
    return np.stack([np.roll(x,tuple(rng.integers(-max_shift,max_shift+1,2)),axis=(-2,-1)) for x in X])

class CNN(nn.Module):
    def __init__(self,padding='circular',channels=6,stride_pool=True):
        super().__init__();c=channels
        layers=[nn.Conv2d(1,c,3,padding=1,padding_mode=padding),nn.ReLU(),
                nn.Conv2d(c,c,3,padding=1,padding_mode=padding),nn.ReLU()]
        if stride_pool:layers.append(nn.AvgPool2d(2))
        layers += [nn.Conv2d(c,2*c,3,padding=1,padding_mode=padding),nn.ReLU()]
        self.features=nn.Sequential(*layers);self.head=nn.Linear(2*c,1)
    def forward(self,x):return self.head(self.features(x).mean((-2,-1)))

def flat_mlp(grid=32,width=48):
    return nn.Sequential(nn.Flatten(),nn.Linear(grid*grid,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1))

def sample_elliptic(n,p,q,rng,noise=.015,x_max=2.5):
    """Amostragem uniforme em x, NÃO em comprimento; janela de x finita.

    Corrige o contador do original: contamos pontos, não blocos anexados.
    """
    pts=[];total=0
    for _ in range(1000):
        x=rng.uniform(-x_max,x_max,max(4*n,100));v=x**3+p*x+q;x=x[v>=0]
        if len(x):
            y=np.sqrt(np.maximum(0,x**3+p*x+q))*rng.choice([-1,1],len(x))
            C=np.column_stack((x,y));pts.append(C);total+=len(C)
        if total>=n:
            P=np.concatenate(pts)[:n];return P+noise*rng.normal(size=P.shape)
    raise RuntimeError('Não foi possível amostrar o locus real na janela escolhida.')

def exact_components(p,q):
    disc=4*p**3+27*q*q
    return np.nan if abs(disc)<1e-9 else (2 if disc<0 else 1)

def knn_adjacency(X,k):
    n=len(X);ix=cKDTree(X).query(X,k=k+1)[1][:,1:]
    W=sparse.coo_matrix((np.ones(n*k),(np.repeat(np.arange(n),k),ix.ravel())),shape=(n,n)).tocsr()
    return W.maximum(W.T)

def knn_overlap(Dhi,Dlo,k=12):
    # Isto é overlap/recall de vizinhos, não o trustworthiness ponderado por ranks.
    a=np.argsort(Dhi,axis=1)[:,1:k+1];b=np.argsort(Dlo,axis=1)[:,1:k+1]
    return np.mean([len(set(x)&set(y))/k for x,y in zip(a,b)])

def embed_metrics(Y,Dintrinsic,k=12):
    Dlo=distance.squareform(distance.pdist(Y));iu=np.triu_indices(len(Y),1)
    near=np.argsort(Dintrinsic,axis=1)[:,1:k+1]
    local=Dlo[np.arange(len(Y))[:,None],near]
    # Fração de arestas locais muito esticadas: diagnóstico, não prova de topologia.
    scale=np.median(local);tear=float(np.mean(local>3*max(scale,1e-12)))
    return dict(knn_preservado=knn_overlap(Dintrinsic,Dlo,k),
                correlacao_global=np.corrcoef(Dintrinsic[iu],Dlo[iu])[0,1],
                fracao_arestas_estiradas=tear)
