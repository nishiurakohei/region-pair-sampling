"""Small frozen-encoder linear probes for main, beta1-aux, and pair-aux models."""
import json
import sys
from pathlib import Path

import numpy as np

P3=Path(__file__).resolve().parent; ROOT=P3.parent
sys.path.insert(0,str(ROOT/"src"))
from dataset import MazeData
from pair_models import build
from pairs import ensure_component_cache, ensure_fixed_pairs, sample_pairs

OUT=ROOT/"results_phase3"/"multitask"


def tag(x): return str(x).replace(".","p")


def load_encoder(model_name,source,lam=None,k=None,seed=1):
    model=build(model_name,np.random.default_rng(1000*seed+(0 if model_name=="cnn" else 500)))
    if source=="main": path=ROOT/"results_hard_maze"/f"train_H3_{model_name}_s{seed}.npz"
    elif source=="beta1": path=ROOT/"results_phase2"/"multitask"/"checkpoints"/f"train_{model_name}_true_l{tag(lam)}_s{seed}.npz"
    else: path=OUT/"checkpoints"/f"train_{model_name}_true_l{tag(lam)}_k{k}_s{seed}.npz"
    z=np.load(path,allow_pickle=False)
    for name,value in model.encoder_params().items(): value[...] = z[f"BEST::{name}"]
    model.load_bn_state({name[8:]:z[name] for name in z.files if name.startswith("BESTBN::")})
    return model


def extract(model,X,pairs=None,bs=256):
    glob=[]; pair=[]
    for s in range(0,len(X),bs):
        q=None if pairs is None else pairs[s:s+bs]
        model.forward(X[s:s+bs],q,train=False); glob.append(model.global_h.copy())
        if q is not None:
            g=np.broadcast_to(model.global_h[:,None],model.hu.shape)
            pair.append(np.concatenate((model.hu,model.hv,np.abs(model.hu-model.hv),model.hu*model.hv,g),axis=2).reshape(-1,5*model.hu.shape[2]))
    return np.concatenate(glob),None if pairs is None else np.concatenate(pair)


def normalize(train,test):
    mu=train.mean(0); sd=train.std(0); sd[sd<1e-6]=1
    return (train-mu)/sd,(test-mu)/sd


def ridge(train_x,train_y,test_x,test_y,alpha=1):
    x,xt=normalize(train_x,test_x); x=np.column_stack((x,np.ones(len(x)))); xt=np.column_stack((xt,np.ones(len(xt))))
    reg=np.eye(x.shape[1]); reg[-1,-1]=0
    w=np.linalg.solve(x.T@x+alpha*reg,x.T@train_y); pred=xt@w
    return {"r2":float(1-np.sum((pred-test_y)**2)/np.sum((test_y-test_y.mean())**2)),
            "rmse":float(np.sqrt(np.mean((pred-test_y)**2)))}


def logistic(train_x,train_y,test_x,test_y,steps=150):
    x,xt=normalize(train_x,test_x); x=np.column_stack((x,np.ones(len(x)))); xt=np.column_stack((xt,np.ones(len(xt))))
    w=np.zeros(x.shape[1]); lr=.2
    for _ in range(steps):
        p=1/(1+np.exp(-np.clip(x@w,-30,30))); g=x.T@(p-train_y)/len(x); g[:-1]+=1e-4*w[:-1]; w-=lr*g
    p=1/(1+np.exp(-np.clip(xt@w,-30,30))); pred=p>.5
    return {"accuracy":float(np.mean(pred==test_y)),"balanced_accuracy":float((np.mean(pred[test_y==1])+np.mean(~pred[test_y==0]))/2)}


def main():
    cfg=json.loads((P3/"config.json").read_text()); ks=json.loads((OUT/"k_selection.json").read_text()); k=ks["selected_k"]
    metrics=json.loads((OUT/"metrics.json").read_text()); p2=json.loads((ROOT/"results_phase2"/"multitask"/"metrics.json").read_text())
    tr,te=MazeData("H3","train"),MazeData("H3","test"); Xtr,Xte=tr.raster_all(),te.raster_all()
    train_pairs,train_target,_=sample_pairs(ensure_component_cache("train"),4,cfg["pair_seed"]+900,cfg["d_min"],0)
    fixed=ensure_fixed_pairs("test",max(cfg["k_values"]),cfg["pair_seed"]+2,cfg["d_min"],True)
    test_pairs,test_target=fixed["pairs"][:,:4],fixed["targets"][:,:4]
    out={"scope":"seed 1 frozen encoders; train-split linear probes evaluated on test","models":{}}
    for model_name in cfg["models"]:
        out["models"][model_name]={}
        settings=(("main",None),("beta1",p2["lambda_selection"]["selected"][model_name]),
                  ("random_pair",metrics["lambda_selection"]["selected"][model_name]))
        for source,lam in settings:
            model=load_encoder(model_name,source,lam,k,1)
            gh_tr,pf_tr=extract(model,Xtr,train_pairs); gh_te,pf_te=extract(model,Xte,test_pairs)
            n=min(16000,len(pf_tr)); idx=np.random.default_rng(7).choice(len(pf_tr),n,replace=False)
            out["models"][model_name][source]={
                "beta1":ridge(gh_tr,tr.arrays["beta1"].astype(float),gh_te,te.arrays["beta1"].astype(float)),
                "reachability":logistic(gh_tr,tr.y.astype(int),gh_te,te.y.astype(int)),
                "random_pair_connectivity":logistic(pf_tr[idx],train_target.ravel()[idx],pf_te,test_target.ravel()),
            }
    (OUT/"representation_probes.json").write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))


if __name__=="__main__": main()
