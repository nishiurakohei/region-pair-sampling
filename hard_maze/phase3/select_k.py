"""Select K from the CNN pilot using validation main BCE only."""
import json
from pathlib import Path

import numpy as np

P3=Path(__file__).resolve().parent; ROOT=P3.parent; OUT=ROOT/"results_phase3"/"multitask"


def tag(x): return str(x).replace(".","p")


def main():
    cfg=json.loads((P3/"config.json").read_text()); lam=cfg["k_pilot_lambda"]
    values={}
    for k in cfg["k_values"]:
        per=[]
        for seed in cfg["aux_only_seeds"]:
            z=np.load(OUT/f"pred_cnn_true_l{tag(lam)}_k{k}_s{seed}.npz",allow_pickle=False)
            h=json.loads(str(z["history"])); per.append(min(x["val_main_bce"] for x in h))
        values[str(k)]={"mean_best_val_main_bce":float(np.mean(per)),"std":float(np.std(per)),"per_seed":per}
    selected=min(cfg["k_values"],key=lambda k:values[str(k)]["mean_best_val_main_bce"])
    out={"criterion":"lowest mean per-seed best validation main BCE","pilot_model":"cnn",
         "pilot_lambda":lam,"seeds":cfg["aux_only_seeds"],"values":values,"selected_k":selected}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"k_selection.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))


if __name__=="__main__": main()
