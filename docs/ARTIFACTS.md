# Large artifact manifest

Large artifacts are intentionally excluded from ordinary Git. Paths below are repository-relative expected locations.

| Artifact | SHA-256 | Associated config | Origin / storage |
|---|---|---|---|
| `hard_maze/external_validation/upstream/maze-extrapolation/models/dt_net.pth` | `b8beed72aad07b300e178f74f0c3628886cfb28fca5d271c451496af38478216` | Stage 0 configs | `mines-opt-ml/maze-extrapolation@6d09263fdb8d61274e4b0e046e07ed929b35af12`; restored by cloning upstream |
| `hard_maze/external_validation/data/easy-to-hard-data/maze_data_train_9.tar.gz` | `1d403b8575ff3e5fc28be7ab3982ca170bc566a2e92a3c50b4e077389c23faca` | `stage0_5_training.json`, `stage1_aux.json` | `https://cs.umd.edu/~tomg/download/Easy_to_Hard_Datav2/maze_data_train_9.tar.gz`; local/artifact storage |
| `hard_maze/external_validation/data/easy-to-hard-data/maze_data_test_9.tar.gz` | `eb5ebc8cbe23a6ad92d4a8a573ca86b83ec524e5c869e776cbb773a2f0b324a5` | Stage 1 held-out evaluation | `https://cs.umd.edu/~tomg/download/Easy_to_Hard_Datav2/maze_data_test_9.tar.gz`; local/artifact storage |
| `hard_maze/results_external/stage0_5_training/checkpoints/seed_101/best_torch.pt` | `a18d84557f2c79a7c605dec348a1324614624387530856a29bdfc5124135a12d` | `stage0_5_resource.json` | Local resource-bounded run; not required to resume full training |
| `hard_maze/results_external/stage0_5_training/checkpoints/seed_102/best_torch.pt` | `8982ac533dd3f15cb51ef853dda813e8b041e67d13ce02992d750f3327a0cfb1` | `stage0_5_resource.json` | Local resource-bounded run |
| `hard_maze/results_external/stage0_5_training/checkpoints/seed_103/best_torch.pt` | `6e83c1b1cc0d1bffe02eff74f192079869662690a60a1271112c73f52ed4e94c` | `stage0_5_resource.json` | Local resource-bounded run |
| `hard_maze/results_external/stage1_aux_learnability/checkpoints/seed_401/best.safetensors` | `00a000cb7f01baca19a425b628d345b15465b89fc1023d81e9b9cbf119dc4981` | `stage1_aux.json` | Local Stage 1 run |
| `hard_maze/results_external/stage0_5_full/checkpoints/seed_*/epoch_*.pt` | generated; record after run | `stage0_5_training.json` | GPU machine artifact storage; epochs 1,2,5,10,20,30,40,50 |
| `hard_maze/results_external/stage0_5_full/checkpoints/seed_*/best.pt` | generated; record after run | `stage0_5_training.json` | GPU machine artifact storage; selected on validation only |

For each new checkpoint, append filename, SHA-256, config path, source commit, model/dataset/shuffle/progressive seeds, and durable storage location. Git LFS is a future option if a shared artifact service is unavailable; it is not configured in this handoff.
