from utils.utils import ConfigClass
import pickle
root = ConfigClass.codeflaws_data_path
test_verdict = pickle.load(open(ConfigClass.codeflaws_test_verdict_pickle, 'rb'))


def key2test_verdict(key):
    info = key.split("-")
    return test_verdict["{}-{}".format(info[0], info[1])][info[3]]


def key2bug(key):
    info = key.split("-")
    return f"{info[0]}-{info[1]}-{info[3]}"


def key2fix(key):
    info = key.split("-")
    return f"{info[0]}-{info[1]}-{info[4]}"


def key2bugfile(key):
    bug = key2bug(key)
    return f"{root}/{key}/{bug}.c"


def key2fixfile(key):
    bug = key2fix(key)
    return f"{root}/{key}/{bug}.c"


def get_gcov_file(key, test):
    return "{}/{}/{}.gcov".format(
        ConfigClass.codeflaws_data_path, key, test)

