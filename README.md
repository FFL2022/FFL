# GNN4FL
## Enviroment
- Server 202.191.56.67: 
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
python3 -m unittest.test_nx_graph_build.py
```
