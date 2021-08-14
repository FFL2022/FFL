from utils.utils import ConfigClass
import pickle
root = ConfigClass.nbl_source_path
test_verdict = pickle.load(open(ConfigClass.nbl_test_verdict_pickle,
                                'rb'))


def key2test_verdict(key):
    info = key.split("-")
    # problem, user, version
    return test_verdict[info[0]][info[2]].keys()

def key2bug(key):
    info = key.split("-")

def get_gcov_file(key, test):
    info = key.split("-")
    return f"{ConfigClass.nbl_test_path}/{info[0]}/{test}-{info[2]}.gcov"
