from torch.utils.data import Dataset
from utils.data_utils import NxDataloader, AstGraphMetadata


class IPyCPyGStatementDataset(Dataset):
    def __init__(self, dataloader: NxDataloader,
                 meta_data: AstGraphMetadata,
                 ast_enc=None):
        self.dataloader = dataloader
        self.meta_data = meta_data
