def make_nbl_dict(key, test_verdict):
    problem_id, uid, program_id = key.split("-")
    nbl = {}
    nbl['problem_id'] = problem_id
    nbl['uid'] = uid
    nbl['program_id'] = program_id
    nbl['test_verdict'] = test_verdict[problem_id][int(program_id)]
    return nbl

