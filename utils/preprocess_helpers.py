import os

def make_dir_if_not_exists(path):
    try:
        os.makedirs(path)
    except:
        pass

def write_to_file(x, filepath):
    with open(filepath, "w") as f:
        f.write(x)

def get_coverage(filename, nline_removed):
    
    def process_line(line):
        tag, line_no, code = line.strip().split(':', 2)
        return tag.strip(), int(line_no.strip()), code
    
    coverage = {}
    with open(filename, "r") as f:
        gcov_file = f.read()
        for idx, line in enumerate(gcov_file.split('\n')):
            if idx <= 4 or len(line.strip()) == 0:
                continue
            
            try:
                tag, line_no, code = process_line(line)
            except:
                print('idx:', idx, 'line:', line)
                print(line.strip().split(':', 2))
                raise
            assert idx!=5 or line_no==1, gcov_file
        
            if tag == '-':
                continue
            elif tag == '#####':
                coverage[line_no - nline_removed] = 0
            else:  
                tag = int(tag) 
                coverage[line_no - nline_removed] = 1
        return coverage

def remove_lib(filename):
    count = 0
    with open(filename, "r") as f:
        with open("temp.c", "w") as t:
            for line in f:
                if line[0] != "#":
                    t.write(line)
                else:
                    count += 1
    return count

def split_data(root="/home/thanhlc/thanhlc/Data/nbl_dataset/"):
    dataset = root + 'dataset.db'
    destination = 'result/network_inputs/bugloc-%s/' % time.strftime("%d-%m")
    mkdir(destination)
    print(destination)

    eval_set = np.load(root + 'data/eval_set.npy', allow_pickle=True).item()
    eval_dict = {}
    for problem_id in eval_set:
        for program_id, row in eval_set[problem_id].items():
            eval_dict[program_id] = row

    eval_set_program_ids = set(eval_dict.keys())
    for id in eval_set_program_ids:
        print(id)
        print(eval_dict[id])
        break
    # print(eval_set_program_ids.get(0))
    # print(eval_dict[eval_set_program_ids[0]])
    print('len(eval_set_program_ids):', len(eval_set_program_ids)) 

    query='''SELECT p.program_id, program, test_id, t.verdict FROM 
        programs p INNER JOIN orgsource o ON o.program_id = p.program_id
        INNER JOIN problems q ON o.problem_id = q.problem_id 
        INNER JOIN test_runs t ON t.program_id = p.program_id
        INNER JOIN test_run_summary trs ON trs.program_id = p.program_id
        WHERE trs.verdict<>"ALL_FAIL" AND q.problem_id=?;'''

    max_programs_per_test_case_result = 700

    test_wise_counts = {}
    eval_test_wise_counts = {}
    all_data = {}
    all_eval_data = {}
    test_id_to_problem_id_map = {}

    with sqlite3.connect(dataset) as conn:
        c = conn.cursor()
    problem_ids = [str(row[0]) for row in c.execute('SELECT DISTINCT problem_id FROM orgsource;')]
    print('len(problem_ids)', len(problem_ids), problem_ids[0])
    
    for problem_id in problem_ids:
            all_data[problem_id] = {}
            for row in c.execute(query, (problem_id,)):
                program_id = row[0]
                program = row[1]
                test_id = str(row[2])
                verdict = row[3]
                if program_id in eval_set_program_ids:
                    try:
                        all_eval_data[program_id]["tests"].append(test_id)
                        all_eval_data[program_id]["verdict"].append(verdict)
                    except KeyError:
                        all_eval_data[program_id] = {}
                        all_eval_data[program_id]["tests"] = [test_id]
                        all_eval_data[program_id]["verdict"] = [verdict]
                else:
                    try:
                        all_data[problem_id][program_id]["tests"].append(test_id)
                        all_data[problem_id][program_id]["verdict"].append(verdict)
                    except KeyError:
                        all_data[problem_id][program_id] = {}
                        all_data[problem_id][program_id]["tests"] = [test_id]
                        all_data[problem_id][program_id]["verdict"] = [verdict]
    total = 0
    for problem_id in all_data:
        for program_id in all_data[problem_id]:
            total += 1
    print('total training data', total)
    c.close()
    conn.close()

    return all_data, all_eval_data  

def find_bug_localization():
    pass

def get_correct_version(program_id):
    pass