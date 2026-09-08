"""Testes rápidos independentes dos experimentos longos.

Executar na raiz: python -m unittest discover -s tests -v
"""
import unittest
import numpy as np
import torch
from scipy import sparse
from scipy.spatial import distance
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from estudos.comum import sphere, fibonacci, geodesic, real_harmonics, poly, curves, CNN, predict, exact_components
from estudos.conjuntos import DeepSets, SetAttention, ellipse_clouds, reorder_clouds

class GeometriaTest(unittest.TestCase):
    def setUp(self):self.rng=np.random.default_rng(845)
    def test_amostragem_norma(self):
        for d in [3,4,100]:
            X=sphere(300,d,self.rng)
            np.testing.assert_allclose(np.linalg.norm(X,axis=1),1,atol=1e-12)
    def test_semente(self):
        np.testing.assert_array_equal(sphere(10,3,np.random.default_rng(1)),sphere(10,3,np.random.default_rng(1)))
    def test_esfera_distancias(self):
        X=sphere(120,3,self.rng);D=geodesic(X);C=distance.squareform(distance.pdist(X))
        np.testing.assert_allclose(C,2*np.sin(D/2),atol=1e-7)
        np.testing.assert_array_equal(np.argsort(C,axis=1),np.argsort(D,axis=1))
    def test_harmonicos(self):
        X=fibonacci(5000);F=real_harmonics(X,3)
        np.testing.assert_allclose(F.T@F/len(F),np.eye(16),atol=.002)
    def test_dimensao_polinomios_restritos(self):
        X=sphere(200,3,self.rng);F=poly(X,3)
        self.assertEqual(F.shape[1],20)
        self.assertEqual(np.linalg.matrix_rank(F),16)
    def test_area_polar(self):
        D=curves(30,self.rng,ntheta=4096)
        np.testing.assert_allclose(np.pi*np.mean(D['r']**2,axis=1),D['area'],atol=1e-12)
    def test_discriminante(self):
        self.assertEqual(exact_components(-3,1),2)
        self.assertEqual(exact_components(-3,3),1)
        self.assertTrue(np.isnan(exact_components(-3,2)))
    def test_h0_mst(self):
        X=self.rng.uniform(size=(30,2));D=distance.squareform(distance.pdist(X))
        deaths=minimum_spanning_tree(D).data
        for eps in [.1,.2,.3,.5]:
            G=sparse.csr_matrix((D<=eps)&(D>0))
            nc=connected_components(G,directed=False,return_labels=False)
            self.assertEqual(nc,1+np.sum(deaths>eps))

class SimetriasTest(unittest.TestCase):
    def test_cnn_circular_sem_subamostragem(self):
        torch.manual_seed(2);m=CNN(stride_pool=False)
        X=np.random.default_rng(2).normal(size=(3,1,32,32)).astype(np.float32)
        np.testing.assert_allclose(predict(m,X),predict(m,np.roll(X,3,axis=-1)),atol=2e-6)
    def test_modelos_de_conjuntos(self):
        rng=np.random.default_rng(3);X,*_=ellipse_clouds(8,rng);Xp=reorder_clouds(X,rng)
        for model in [DeepSets(),SetAttention(),SetAttention(layers=2)]:
            np.testing.assert_allclose(predict(model,X),predict(model,Xp),atol=2e-6)
    def test_mascaras_ignoram_padding(self):
        rng=np.random.default_rng(7);X,*_=ellipse_clouds(8,rng);Y=X.copy()
        Y[:,:,:2][Y[:,:,2]==0]=1000
        for model in [DeepSets(),SetAttention()]:
            np.testing.assert_allclose(predict(model,X),predict(model,Y),atol=2e-6)
    def test_reversao_area(self):
        rng=np.random.default_rng(11);X,_,_,A,_=ellipse_clouds(10,rng);Z=reorder_clouds(X,reverse=True)
        B=[]
        for z in Z:
            n=int(z[:,2].sum());p=z[:n,:2]
            B.append(.5*np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))
        np.testing.assert_allclose(B,-A,atol=1e-6)

if __name__=='__main__':unittest.main()
