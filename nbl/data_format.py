from utils.utils import ConfigClass
import pickle
root = ConfigClass.nbl_source_path
test_verdict = pickle.load(open(ConfigClass.nbl_test_verdict_pickle,
                                'rb'))


def key2test_verdict(key):
    pid, uid, vid = key.split("-")
    # problem, user, version
    return test_verdict[pid][vid].keys()


def get_gcov_file(key, test):
    pid, uid, vid = key.split("-")
    return f"{ConfigClass.nbl_test_path}/{pid}/{test}-{vid}.gcov"
