def get_coverage(filename):
    
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
                coverage[line_no] = 0
            else:  
                tag = int(tag) 
                coverage[line_no] = 1
            
        return coverage


print(get_coverage("examples/sources/3024/tests/IN_40013.txt-code.gcov")) 
