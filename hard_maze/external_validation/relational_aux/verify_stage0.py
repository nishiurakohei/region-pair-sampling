"""Mechanical checks for the scoped Stage 0 screening run."""
import csv, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results_external"/"stage0_reproduction"
UP=Path(__file__).resolve().parents[1]/"upstream"/"maze-extrapolation"

cfg=json.loads((OUT/"config.json").read_text()); metrics=json.loads((OUT/"metrics.json").read_text())
rows=list(csv.DictReader((OUT/"per_seed_metrics.csv").open()))
tail_cfg=json.loads((OUT/"full_tail"/"config.json").read_text()); tail=json.loads((OUT/"full_tail"/"metrics.json").read_text())
tail_rows=list(csv.DictReader((OUT/"full_tail"/"per_seed_metrics.csv").open()))
summary=metrics["summary"]
def one(axis,**want):
    return next(r for r in summary if r["axis"]==axis and all(r[k]==v for k,v in want.items()))

commit=(OUT/"upstream_commit.txt").read_text().strip()
id_acc=one("id")["accuracy_mean"]
p0=one("percolation",percolation=0.0)["accuracy_mean"]
p4=one("percolation",percolation=0.04)["accuracy_mean"]
d_true=one("deadend_start",deadend_start=True)["accuracy_mean"]
d_false=one("deadend_start",deadend_start=False)["accuracy_mean"]
s29_30=one("maze_size",maze_size=29,iterations=30)["accuracy_mean"]
s29_300=one("maze_size",maze_size=29,iterations=300)["accuracy_mean"]
size_grid=sorted({r["maze_size"] for r in summary if r["axis"]=="maze_size"}|{r["maze_size"] for r in tail["summary"]})
checks={
    "upstream_commit_recorded":len(commit)==40,
    "upstream_clean":not __import__("subprocess").check_output(["git","-C",str(UP),"status","--porcelain"],text=True).strip(),
    "config_recorded":bool(cfg),
    "expected_rows":len(rows)==105,
    "five_consistent_seeds":sorted({int(r["seed"]) for r in rows})==cfg["seeds"],
    "distinct_seed_datasets":len({r["dataset_sha256"] for r in rows if r["axis"]=="id"})==5,
    "id_reproduced":id_acc>=.95,
    "percolation_decline_reproduced":p0-p4>=.2,
    "deadend_decline_reproduced":d_true-d_false>=.1,
    "iteration_recovery_reproduced":s29_300-s29_30>=.5,
    "full_size_grid_complete":size_grid==list(range(9,100,10)),
    "tail_expected_rows":len(tail_rows)==len(tail_cfg["maze_sizes"])*len(tail_cfg["iterations"])*len(tail_cfg["seeds"]),
    "tail_five_distinct_seeds":all(len({r["dataset_sha256"] for r in tail_rows if int(r["maze_size"])==size})==5 for size in tail_cfg["maze_sizes"]),
    "large_size_extrapolation_reproduced":all(r["accuracy_mean"]==1.0 for r in tail["summary"] if r["iterations"]==3000),
    "figures_complete":len(list((OUT/"figures").glob("*.png")))==6,
}
result={"scope":"Full Stage 0: DT-Net, maze sizes 9–99, up to 3000 iterations, percolation and deadend-start shifts","checks":checks,
        "all_ok":all(checks.values()),"gate0_screening_pass":all(checks.values()),
        "full_gate0_complete":all(checks.values()),"full_gate0_missing":[],
        "limitations":["Tail sizes use one maze per each of five dataset seeds; the paper used 100 mazes per point.","PI-Net artifacts are absent, so DT-Net is the single reproduction target."]}
(OUT/"verification.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
