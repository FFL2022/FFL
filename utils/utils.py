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

    nbl_test_verdict_pickle = "data_nbl/test_verdict.pkl"
    nbl_source_path = "data_nbl/data/sources"
    nbl_test_path = "data_nbl/data/tests"
    os.makedirs(nbl_source_path, exist_ok=True)

    codeflaws_data_path = "data_codeflaws/data"

    # raw dir
    nbl_raw_dir = "data_nbl/"

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
    preprocess_dir_codeflaws = "./preprocessed/codeflaws"
    preprocess_dir_nbl= "./preprocessed/nbl"

    trained_dir_codeflaws = './trained/codeflaws/{}'.format(today)
    result_dir_codeflaws = './result/codeflaws/{}'.format(today)

    trained_dir_nbl = './trained/nbl/{}'.format(today)
    result_dir_nbl = './result/nbl/{}'.format(today)

    trained_dir_nbl_a = './trained/nbl/{}'.format(today)

    os.makedirs(result_dir_nbl, exist_ok=True)
    os.makedirs(trained_dir_nbl, exist_ok=True)
    os.makedirs(result_dir_codeflaws, exist_ok=True)
    os.makedirs(trained_dir_codeflaws, exist_ok=True)

    os.makedirs(preprocess_dir_nbl, exist_ok=True)
    os.makedirs(preprocess_dir_codeflaws, exist_ok=True)

    n_epochs = 100
    print_rate = 1
    save_rate = 1

    cfg_label_corpus = ["entry_node", "COMMON", "IF", "ELSE",
                        "ELSE_IF", "END_IF", "FOR", "WHILE",
                        "DO_WHILE", "PSEUDO", "CALL", "END"]
    ast_cpp_command = "java -jar ./jars/ast_extractor_cpp.jar "
    ast_java_command = "java -jar ./jars/ast_extractor_java.jar "

    def check_is_stmt_java(ntype):
        return 'Statement' in ntype

    def check_is_stmt_cpp(ntype):
        return ntype in ['for', 'while', 'switch', 'decl_stmt',
                         'if_stmt', 'case', 'else', 'break', 'do',
                         'continue', 'goto', 'empty_stmt', 'expr_stmt',
                         'default', 'label', 'continue', 'return',
                         'placeholder_stmt']

ConfigClass = ConfigClassDat
