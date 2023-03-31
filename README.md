# SGNN
 Node-based subgraph GNN and localized 2-FWL.

### Requirements
```
python=3.8
torch=1.11.0
PyG=2.1.0
pytorch_lightning=1.9.4
wandb=0.13.11
```

To run experiments for different model:
```
python train_zinc.py --config_file=configs/zinc_slfwl_egonetsde.yaml
python train_zinc.py --config_file=configs/zinc_lfwl_egonetsde.yaml
python train_zinc.py --config_file=configs/zinc_sswl+_egonetsde.yaml
```
