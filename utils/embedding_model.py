import fasttext
from utils.utils import ConfigClass
embedding_model = fasttext.load_model(ConfigClass.pretrained_fastext)
