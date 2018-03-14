import numpy as np

def load_dataset(file_name):
    char_vocab = set([])
    tag_vocab = set([])
    dataset = []
    with open(file_name) as f:
        for line in f.readlines():
            example = line.split()
            dataset.append(example)
            char_vocab.update(example[1].lower())
            char_vocab.update(example[3].lower())
            tag_vocab.update(example[0].split(','))
            tag_vocab.update(example[2].split(','))
    char_vocab.update(['<unk>', '<pad>'])
    tag_vocab.add('<unk>')

    return np.array(dataset), sorted(char_vocab), sorted(tag_vocab)

def preprocess_data(dataset, char_vocab, tag_vocab, T_max):
    # TODO
    pass
