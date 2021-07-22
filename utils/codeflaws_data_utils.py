
def make_codeflaws_dict(key, test_verdict):
    info = key.split("-")
    codeflaws = {}
    codeflaws['container'] = key
    codeflaws['c_source'] = "{}-{}-{}".format(
        info[0], info[1], info[3])
    codeflaws['test_verdict'] = test_verdict["{}-{}".format(
        info[0], info[1])][info[3]]
    return codeflaws
