# GNN4FL
## Enviroment
- Server: 
```
conda activate gbl
```
## How to build dgl graph (without node features)
```
python3 dataset.py
```
- Funtion: dataset.build_dgl_graph(...). 
- Return: It will return G (dgl graph of cfg + test + tests), ast_id2idx, cfg_id2idx, tests_id2idx

## Testing Graph Visualization
```bash
python3 -m selfunittest.test_nx_graph_build.py
```
## New guide
### Prerequisite:
- anaconda
- srcml
- java 1.11 + (Version 16+ SDK)
### Download:
Download jars file [here](https://drive.google.com/file/d/1gM1j_sJRhrpcGoJgyd2Hyot-aZqIMRdl/view?usp=sharing) and unzip to the root directory

## Ego Graph Extract
Used parameter is 3 hops
