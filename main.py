import argparse,json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from torchvision import datasets,transforms,utils

def main():
 p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=2); p.add_argument('--limit',type=int,default=5000); a=p.parse_args(); r=Path(__file__).parent; d=r/'data'; (r/'outputs').mkdir(exist_ok=True); d.mkdir(exist_ok=True)
 ds=datasets.FashionMNIST(d,train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x:x*2-1)])); ds=Subset(ds,range(min(a.limit,len(ds)))); dl=DataLoader(ds,batch_size=64,shuffle=True)
 device='cuda' if torch.cuda.is_available() else 'cpu'; steps=100; beta=torch.linspace(1e-4,.02,steps,device=device); alpha=1-beta; abar=alpha.cumprod(0)
 class Denoiser(nn.Module):
  def __init__(self):
   super().__init__(); self.time=nn.Embedding(steps,32); self.net=nn.Sequential(nn.Conv2d(33,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,64,3,padding=1),nn.SiLU(),nn.Conv2d(64,64,3,padding=1),nn.SiLU(),nn.Conv2d(64,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,1,3,padding=1))
  def forward(self,x,t):
   te=self.time(t)[:,:,None,None].expand(-1,-1,28,28); return self.net(torch.cat([x,te],1))
 model=Denoiser().to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3); hist=[]
 for e in range(a.epochs):
  model.train(); losses=[]
  for x,_ in dl:
   x=x.to(device); t=torch.randint(0,steps,(len(x),),device=device); noise=torch.randn_like(x); xt=abar[t,None,None,None].sqrt()*x+(1-abar[t,None,None,None]).sqrt()*noise; pred=model(xt,t); loss=nn.functional.mse_loss(pred,noise); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
  hist.append({'epoch':e+1,'train_noise_mse':round(sum(losses)/len(losses),4)}); print(hist[-1])
 model.eval(); x=torch.randn(16,1,28,28,device=device)
 with torch.no_grad():
  for i in reversed(range(steps)):
   t=torch.full((len(x),),i,device=device,dtype=torch.long); x=(x-beta[i]/(1-abar[i]).sqrt()*model(x,t))/alpha[i].sqrt()
   if i: x+=beta[i].sqrt()*torch.randn_like(x)
 utils.save_image((x.clamp(-1,1)+1)/2,r/'outputs/samples.png',nrow=4); torch.save(model.state_dict(),r/'outputs/ddpm.pt'); (r/'outputs/metrics.json').write_text(json.dumps(hist,indent=2)); print('Generated outputs/samples.png')
if __name__=='__main__': main()
