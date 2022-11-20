from torch.utils.data import Dataset
from codeflaws.dataloader_cfl import CodeflawsCFLNxStatementDataset, \
    ASTMetadata
from utils.utils import ConfigClass
from Typing import List
import os

class CodeflawsCFLPyGStatementDataset(Dataset):
    def __init__(self, nx_dataset: CodeflawsCFLNxStatementDataset,
                 idxs: List[int],
                 meta_data: ASTMetadata,
                 save_dir=ConfigClass.preprocess_dir_codeflaws):
        self.nx_dataset = nx_dataset
        self.meta_data = meta_data if meta_data else ASTMetadata(nx_dataset)
        self.save_dir = save_dir
        self.vocab_dict = dict(tuple(line.split()) for line in open(
            'preprocess/codeflaws_vocab.txt', 'r'))
        self.graph_save_path = f"{save_dir}/pyg_cfl_stmt.pkl"
        self
        if self.has_cache():
            self.load()
        else:
            self.process()
            self.save()

        def has_cache(self):
            return os.path.exists
