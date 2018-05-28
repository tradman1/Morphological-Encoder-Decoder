from __future__ import print_function

def make_preds_file(file_name, suff):
    preds = []
    with open(file_name, 'r') as f:
        for line in f:
            preds.append("".join(line.split()))
    with open(file_name+suff, 'a') as f:
        for line in preds:
            print(line, file=f)

make_preds_file("predictions/pred-russian-embed500", "-ep20")
