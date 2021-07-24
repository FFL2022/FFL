import os
from datetime import date
# from train import train


class ConfigClassThanhServer(object):
    pretrained_fastext = '/home/thanhlc/thanhlc/Data/c_pretrained.bin'
    test_verdict_pickle = "/home/thanhlc/thanhlc/Data/nbl_dataset/test_verdict.pkl"
    nbl_source_path = "/home/thanhlc/thanhlc/Data/nbl_dataset/sources"
    nbl_test_path = "/home/thanhlc/thanhlc/Data/nbl_dataset/data/tests"
    codeflaws_data_path = "/home/minhld/codeflaws/data"

    # raw dir
    raw_dir = "/home/thanhlc/thanhlc/Data/nbl_dataset"

    # For training
    train_cfgidx_map_json = "/home/thanhlc/thanhlc/Data/nbl_dataset/training_dat.json"
    train_cfgidx_map_pkl = "/home/thanhlc/thanhlc/Data/nbl_dataset/training_data.pkl"
    eval_cfgidx_map_pkl = "/home/thanhlc/thanhlc/Data/nbl_dataset/eval_data.pkl"
    codeflaws_train_cfgidx_map_pkl = "/home/minhld/codeflaws/training_data.pkl"
    codeflaws_eval_cfgidx_map_pkl = "/home/minhld/codeflaws/eval_data.pkl"
    codeflaws_full_cfgidx_map_pkl = "/home/minhld/codeflaws/full_data.pkl"
    codeflaws_test_verdict_pickle = "/home/minhld/codeflaws/test_verdict.pkl"

    bug_lines_info_pkl = "/home/thanhlc/thanhlc/Data/nbl_dataset/bug_lines_info.pkl"

    today = date.today().strftime("%b-%d-%Y")
    preprocess_dir = "./preprocessed"
    trained_dir = './trained/{}'.format(today)
    result_dir = './result/{}'.format(today)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(trained_dir, exist_ok=True)

    n_epochs = 100
    print_rate = 1
    save_rate = 3

    cfg_label_corpus = ["entry_node", "COMMON", "IF", "ELSE",
                        "ELSE_IF", "END_IF", "FOR", "WHILE",
                        "DO_WHILE", "PSEUDO", "CALL", "END"]


class ConfigClassDat(object):
    pretrained_fastext = 'preprocess/c_pretrained.bin'

    test_verdict_pickle = "data_nbl/nbl_dataset/test_verdict.pkl"
    nbl_source_path = "data_nbl/nbl_dataset/sources"
    nbl_test_path = "data_nbl/nbl_dataset/data/tests"

    codeflaws_data_path = "data_codeflaws/data"

    # raw dir
    raw_dir = "data_nbl/nbl_dataset"

    # For training
    train_cfgidx_map_json = "data_nbl/nbl_dataset/training_dat.json"
    train_cfgidx_map_pkl = "data_nbl/nbl_dataset/training_data.pkl"
    eval_cfgidx_map_pkl = "data_nbl/nbl_dataset/eval_data.pkl"

    codeflaws_train_cfgidx_map_pkl = "data_codeflaws/training_data.pkl"
    codeflaws_eval_cfgidx_map_pkl = "data_codeflaws/eval_data.pkl"
    codeflaws_test_verdict_pickle = "data_codeflaws/test_verdict.pkl"
    codeflaws_full_cfgidx_map_pkl = "data_codeflaws/full_data.pkl"
    codeflaws_all_keys = "data_codeflaws/all_keys.pkl"

    bug_lines_info_pkl = "data_nbl/nbl_dataset/bug_lines_info.pkl"

    today = date.today().strftime("%b-%d-%Y")
    preprocess_dir = "./preprocessed"
    trained_dir = './trained/{}'.format(today)
    result_dir = './result/{}'.format(today)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(trained_dir, exist_ok=True)

    n_epochs = 100
    print_rate = 1
    save_rate = 3

    cfg_label_corpus = ["entry_node", "COMMON", "IF", "ELSE",
                        "ELSE_IF", "END_IF", "FOR", "WHILE",
                        "DO_WHILE", "PSEUDO", "CALL", "END"]


ConfigClass = ConfigClassDat

if not os.path.exists(ConfigClass.trained_dir):
    os.makedirs(ConfigClass.trained_dir)
