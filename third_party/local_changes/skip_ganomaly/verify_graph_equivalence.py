import copy
from types import SimpleNamespace
import torch
from lib.models.networks import define_G,define_D
from lib.models.skipganomaly import Skipganomaly
from lib.loss import l2_loss
torch.manual_seed(7)
# Small input/width solely for an algebraic regression, not the MVTec experiment.
opt=SimpleNamespace(isize=32,nc=3,ngf=8,ndf=8,nz=100,ngpu=1,gpu_ids=[],w_adv=1.,w_con=50.,w_lat=1.)
m=Skipganomaly.__new__(Skipganomaly)
m.opt=opt
m.netg=define_G(opt)
m.netd=define_D(opt)
refg,refd=copy.deepcopy(m.netg),copy.deepcopy(m.netd)
m.input=torch.randn(4,3,32,32)
m.real_label=torch.ones(4)
m.fake_label=torch.zeros(4)
m.l_adv=torch.nn.BCELoss()
m.l_con=torch.nn.L1Loss()
m.l_lat=l2_loss
m.optimizer_g=torch.optim.Adam(m.netg.parameters(),lr=.0002,betas=(.5,.999))
m.optimizer_d=torch.optim.Adam(m.netd.parameters(),lr=.0002,betas=(.5,.999))
# Official forward and loss dependency structure, gradients evaluated before
# mutation (the old retain_graph + early optimizer step is illegal on torch 2).
fake=refg(m.input)
pr,fr=refd(m.input)
pf,ff=refd(fake)
lat=l2_loss(ff,fr)
lg=m.l_adv(pf,m.real_label)+50*m.l_con(fake,m.input)+lat
gg=torch.autograd.grad(lg,tuple(refg.parameters()),retain_graph=True)
pdf,_=refd(fake.detach())
ld=m.l_adv(pr,m.real_label)+m.l_adv(pdf,m.fake_label)+lat
dg=torch.autograd.grad(ld,tuple(refd.parameters()))
m.optimize_params()
gerr=max(float((a-p.grad).abs().max()) for a,p in zip(gg,m.netg.parameters()))
derr=max(float((a-p.grad).abs().max()) for a,p in zip(dg,m.netd.parameters()))
for a,p in zip(gg,m.netg.parameters()): torch.testing.assert_close(a,p.grad,rtol=1e-4,atol=1e-5)
for a,p in zip(dg,m.netd.parameters()): torch.testing.assert_close(a,p.grad,rtol=1e-4,atol=1e-5)
assert all(torch.equal(a,b) for a,b in zip(refd.buffers(),m.netd.buffers()))
assert all(torch.equal(a,b) for a,b in zip(refg.buffers(),m.netg.buffers()))
print('PASS official-vs-modern gradient equivalence; max absolute G diff',gerr,'D diff',derr,'BN buffers identical')
