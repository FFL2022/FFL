import os

class ConfigClass(object):
    pretrained_fastext = '/home/thanhlc/thanhlc/Data/c_pretrained.bin'
    test_verdict_pickle = "/home/thanhlc/thanhlc/Data/nbl_dataset/test_verdict.pkl"
    nbl_source_path = "/home/thanhlc/thanhlc/Data/nbl_dataset/sources"
    nbl_test_path = "/home/thanhlc/thanhlc/Data/nbl_dataset/data/tests"

    # raw dir
    raw_dir = "/home/thanhlc/thanhlc/Data/nbl_dataset"

    # For training
    train_cfgidx_map_json = "/home/thanhlc/thanhlc/Data/nbl_dataset/training_dat.json"
    train_cfgidx_map_pkl = "/home/thanhlc/thanhlc/Data/nbl_dataset/training_data.pkl"

    bug_lines_info_pkl = "/home/thanhlc/thanhlc/Data/nbl_dataset/bug_lines_info.pkl"

    preprocess_dir = "./preprocessed"
    trained_dir = './trained'

    n_epochs = 100
    print_rate = 5
    save_rate = 5

    cfg_label_corpus = ["entry_node", "COMMON", "IF", "ELSE",
                        "ELSE_IF", "END_IF", "FOR", "WHILE",
                        "DO_WHILE", "PSEUDO", "CALL", "END"]

if not os.path.exists(ConfigClass.trained_dir):
    os.makedirs(ConfigClass.trained_dir)
