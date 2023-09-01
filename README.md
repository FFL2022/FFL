
# FFL
## Dataset Downloads
- Download dataset **Codeflaws** here: https://codeflaws.github.io
- Download dataset **Prutor** here: https://bitbucket.org/iiscseal/nbl/src/master/

Extract the 2 dataset into `data_codeflaws` and `data_nbl` respectively
## Environment: 
The following dependencies are required to train or run the model:
- pytorch (1.6.0+)
- fasttext (0.9.2)
- dgl (1.6.0)
- networkx (2.5.1)


Optional (for visualization):
- pygraphviz and graphviz


Java: version 1.11+
- Gumtree-based annotation package: https://drive.google.com/file/d/1hQ5wmSs2K6dPdhC38VUzn-dtK3J19LDf/view?usp=sharing

Extract `jars.zip` into folder `jars`
## Training: 
```bash
# Codeflaws node-level
python3 -m codeflaws.train_nx_a_nc_old
# NBL node-level
python3 -m nbl.train_nx_a_nc

# Codeflaws statement-level
python3 -m codeflaws.train_nx_astdiff_nocontent_gumtree
# Prutor statement-level
python3 -m nbl.train_nx_astdiff_nocontent_gumtree
```

## Evaluation
The training script already contain evaluation, by disabling `train()` procedure, the script will script directly to evaluation.
Copy the pretrained model into `train_dirs` in the configured `utils/utils.py` for evaluate.
### Pretrained model:
- Codeflaws node-level objective: https://drive.google.com/file/d/1E0XHg5J3wBFaGh8DYL4DmCNf-0z9z58_/view?usp=sharing
- Codeflaws statement-level objective: https://drive.google.com/file/d/1D6_YrUDCfdYqgsRwvpRA-le_MHDrFMhV/view?usp=sharing
- Prutor node-level objective: https://drive.google.com/file/d/1kfL8TpxYA491PUyKvd4nzpbFwcfqGGyY/view?usp=sharing
- NBL statement-level objective: https://drive.google.com/file/d/1Da00veDZb-445eruuE3boOoxzl1twak8/view?usp=sharing


## Others
Please note that while this is not required in our original settings, codebert pretrain file can be placed in `preprocess` folder for each AST content to be used instead of just nodetype.

Please cite the following article if you find FFL to be useful:
```
@article{Nguyen2022,
   author = {Thanh-Dat Nguyen and Thanh Le-Cong and Duc-Minh Luong and Van-Hai Duong and Xuan-Bach D Le and David Lo and Quyet-Thang Huynh},
   city = {Cyprus},
   journal = {The 38th IEEE International Conference on Software Maintenance and Evolution},
   keywords = {Graph Neural Network,Index Terms-Fault Localization,Programming Education},
   month = {11},
   title = {FFL: Fine-grained Fault Localization for Student Programs via Syntactic and Semantic Reasoning},
   url = {https://github.com/FFL2022/FFL},
   year = {2022},
}
```

