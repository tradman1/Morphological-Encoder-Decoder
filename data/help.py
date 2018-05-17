from __future__ import print_function
import numpy as np

def create_result_file(gold_file, guess_file):
    gold_l = []
    guess_l = []
    with open(gold_file, 'r') as gold:
        for line in gold.readlines():
            example = line.split()
            example[3] = example[3].lower()
            gold_l.append(example)
        with open(guess_file, 'r+') as guess:
            for line in guess.readlines():
                guess_l.append(line.strip())
            gold_l = np.array(gold_l)
            guess_l = np.array(guess_l)
        with open(gold_file+"-ready", 'a') as output:
            for i in range(gold_l.shape[0]):
                print(" ".join(gold_l[i,:]), file=output)

            gold_l[:,3] = guess_l
    with open(guess_file+"-ready", 'a') as output:
        for i in range(gold_l.shape[0]):
            print(" ".join(gold_l[i,:]), file=output)



if __name__ == "__main__":
    create_result_file("german-task2-test", "german-task2-test-results2")
