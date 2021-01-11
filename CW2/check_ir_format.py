
# how to use:
# python check_ir_format.py <your_submission_file_here>

import sys

CORRECT_HEADER = "system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20"

def check_file(path, verbose=True):
    correct = True
    seen_system_query_combos = set([])
    with open(path) as infile:
        header = infile.readline().strip()
        if header != CORRECT_HEADER:
            if verbose:
                print("Header is missing or incorrect, it should be (exactly):")
                print(CORRECT_HEADER)
            correct = False
        for i,line in enumerate(infile.readlines()):
            line = line.strip()
            parts = line.split(',')
            if len(parts) != 8:
                if verbose:
                    print(f"Incorrect number of columns in line {i+1}. Expected:8, Got:{len(parts)}")
                correct = False
            else:
                system,query,p,r,rp,m,ndcg10,ndcg20 = line.split(',')
                if (system,query) in seen_system_query_combos:
                    if verbose:
                        print(f"duplicate (system,query) pair on line {i+1}:{(system,query)}")
                    correct = False
                seen_system_query_combos.add( (system,query) )
                if system not in '1 2 3 4 5 6'.split():
                    if verbose:
                        print(f"Invalid system number: {system}")
                    correct = False
                if query not in '1 2 3 4 5 6 7 8 9 10 mean'.split():
                    if verbose:
                        print(f"Invalid query number: {query}")
                    correct = False
                for j,score in enumerate([p,r,rp,m,ndcg10,ndcg20]):
                    try:
                        float_score = float(score)
                    except:
                        if verbose:
                            print(f"Invalid score on line {i+1}, col {j+1}: {score}")
                        correct = False
        for system in '1 2 3 4 5 6'.split():
            for query in '1 2 3 4 5 6 7 8 9 10 mean'.split():
                if (system,query) not in seen_system_query_combos:
                    if verbose:
                        print("Missing (system,query) pair:",(system,query))
                    correct = False
        if correct and verbose:
            print("File is in correct format.")
    return correct

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python check_ir_format.py <your_submission_file>")
    else:
        check_file(sys.argv[1])
