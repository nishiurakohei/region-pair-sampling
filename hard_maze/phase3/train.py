"""Train one auxiliary-only or multi-task random-pair connectivity model."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

P3 = Path(__file__).resolve().parent
ROOT = P3.parent
sys.path.insert(0, str(ROOT / "src"))
from dataset import MazeData
from nn import Adam, bce_with_logits
from pair_models import build
from pairs import ensure_component_cache, ensure_fixed_pairs, sample_pairs


def tag(x): return str(x).replace(".", "p")


def save(path, model, opt, meta, best, best_bn):
    d={f"P::{k}":v for k,v in model.params().items()}
    d.update({f"BN::{k}":v for k,v in model.bn_state().items()})
    d.update({f"OPT::{k}":v for k,v in opt.state().items()})
    d.update({f"BEST::{k}":v for k,v in best.items()})
    d.update({f"BESTBN::{k}":v for k,v in best_bn.items()})
    d["META"]=np.frombuffer(json.dumps(meta).encode(),np.uint8)
    np.savez(path,**d)


def load(path,model,opt):
    z=np.load(path,allow_pickle=False)
    for k,v in model.params().items(): v[...] = z[f"P::{k}"]
    model.load_bn_state({k[4:]:z[k] for k in z.files if k.startswith("BN::")})
    opt.load_state({k[5:]:z[k] for k in z.files if k.startswith("OPT::")})
    return (json.loads(bytes(z["META"]).decode()),
            {k[6:]:z[k] for k in z.files if k.startswith("BEST::")},
            {k[8:]:z[k] for k in z.files if k.startswith("BESTBN::")})


def fixed(split,k,cfg):
    z=ensure_fixed_pairs(split,max(cfg["k_values"]),cfg["pair_seed"]+(1 if split=="val" else 2),
                         cfg["d_min"],distances=True)
    return z["pairs"][:,:k],z["targets"][:,:k]


def evaluate(model,X,y,pairs,target,bs=256):
    main=np.empty(len(y),np.float32); pair=np.empty(target.shape,np.float32)
    for s in range(0,len(y),bs):
        ml,pl=model.forward(X[s:s+bs],pairs[s:s+bs],train=False)
        main[s:s+bs]=ml; pair[s:s+bs]=pl
    mbce,_=bce_with_logits(main,y); pbce,_=bce_with_logits(pair.ravel(),target.ravel())
    return float(mbce),float(pbce),1/(1+np.exp(-np.clip(main,-40,40))),1/(1+np.exp(-np.clip(pair,-40,40)))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("mode",choices=["aux","multi"])
    ap.add_argument("model",choices=["cnn","resnet"])
    ap.add_argument("condition",choices=["true","shuffled"])
    ap.add_argument("lambda_",type=float)
    ap.add_argument("k",type=int)
    ap.add_argument("seed",type=int)
    ap.add_argument("--budget",type=float,default=150)
    a=ap.parse_args()
    if a.mode=="aux" and (a.condition!="true" or a.lambda_!=1):
        raise ValueError("aux mode uses true labels and lambda=1")
    cfg=json.loads((P3/"config.json").read_text())
    base=ROOT/"results_phase3"/("aux_only" if a.mode=="aux" else "multitask")
    name=(f"{a.model}_s{a.seed}" if a.mode=="aux" else
          f"{a.model}_{a.condition}_l{tag(a.lambda_)}_k{a.k}_s{a.seed}")
    pred=base/f"pred_{name}.npz"; ckdir=base/"checkpoints"; ckdir.mkdir(parents=True,exist_ok=True)
    ckpt=ckdir/f"train_{name}.npz"
    if pred.exists(): print("ALREADY DONE",pred); return
    tr,va,te=[MazeData("H3",s) for s in ("train","val","test")]
    Xtr,Xva=tr.raster_all(),va.raster_all()
    labels=ensure_component_cache("train")
    vap,vay=fixed("val",a.k,cfg); tep,tey=fixed("test",a.k,cfg)
    rng=np.random.default_rng(1000*a.seed+(0 if a.model=="cnn" else 500))
    model=build(a.model,rng); opt=Adam(model.params(),lr=cfg["learning_rate"],wd=cfg["weight_decay"])
    if ckpt.exists(): meta,best,best_bn=load(ckpt,model,opt)
    else:
        meta={"mode":a.mode,"model":a.model,"condition":a.condition,"lambda":a.lambda_,"k":a.k,
              "seed":a.seed,"epoch":0,"best_value":1e9,"best_epoch":-1,"patience":0,"history":[],
              "pair_seed":cfg["pair_seed"]+a.seed,"shuffle_seed":cfg["shuffle_seed"]+a.seed}
        best={k:v.copy() for k,v in model.params().items()}; best_bn={k:v.copy() for k,v in model.bn_state().items()}
    t0=time.time(); done=False
    while meta["epoch"]<cfg["max_epochs"] and time.time()-t0<a.budget:
        epoch=meta["epoch"]
        pairs,targets,_=sample_pairs(labels,a.k,cfg["pair_seed"]+a.seed,cfg["d_min"],epoch)
        used=targets
        if a.condition=="shuffled":
            perm=np.random.default_rng(cfg["shuffle_seed"]+a.seed+1000003*epoch).permutation(targets.size)
            used=targets.ravel()[perm].reshape(targets.shape)
        order=np.random.default_rng(9999*a.seed+epoch).permutation(tr.n)
        mt=pt=0.0
        for s in range(0,tr.n,cfg["batch_size"]):
            idx=order[s:s+cfg["batch_size"]]; model.zero_grad()
            ml,pl=model.forward(Xtr[idx],pairs[idx],train=True)
            main_loss,gm=bce_with_logits(ml,tr.y[idx]); pair_loss,gp=bce_with_logits(pl.ravel(),used[idx].ravel())
            if a.mode=="aux": gm=np.zeros_like(gm); weight=1.0
            else: weight=a.lambda_
            model.backward(gm,(weight*gp).reshape(pl.shape)); opt.step(model.grads())
            mt+=main_loss*len(idx); pt+=pair_loss*len(idx)
        meta["epoch"]+=1
        vm,vp,_,_=evaluate(model,Xva,va.y,vap,vay)
        value=vp if a.mode=="aux" else vm
        meta["history"].append({"epoch":meta["epoch"],"train_main_bce":float(mt/tr.n),
                                "train_pair_bce":float(pt/tr.n),"val_main_bce":vm,"val_pair_bce":vp})
        if value<meta["best_value"]-1e-4:
            meta["best_value"],meta["best_epoch"],meta["patience"]=value,meta["epoch"],0
            best={k:v.copy() for k,v in model.params().items()}; best_bn={k:v.copy() for k,v in model.bn_state().items()}
        else: meta["patience"]+=1
        print(f"[{name}] ep{meta['epoch']} main={mt/tr.n:.4f}/{vm:.4f} pair={pt/tr.n:.4f}/{vp:.4f} best={meta['best_value']:.4f}",flush=True)
        done=meta["patience"]>=cfg["patience"] or meta["epoch"]>=cfg["max_epochs"]
        save(ckpt,model,opt,meta,best,best_bn)
        if done: break
    if not done: print("PARTIAL",meta["epoch"]); return
    for k,v in model.params().items(): v[...] = best[k]
    model.load_bn_state(best_bn)
    Xm=te.raster_all(); _,_,main_prob,pair_prob=evaluate(model,Xm,te.y,tep,tey)
    np.savez_compressed(pred,main_prob=main_prob,pair_prob=pair_prob,pair_targets=tey,pairs=tep,
                        y=te.y,pair_id=te.arrays["pair_id"],best_epoch=meta["best_epoch"],
                        history=json.dumps(meta["history"]),pair_seed=meta["pair_seed"],
                        shuffle_seed=meta["shuffle_seed"] if a.condition=="shuffled" else -1)
    print("DONE",pred)


if __name__=="__main__": main()
