import numpy as np

def load_dataset(file_name):
    char_vocab = set([])
    tag_vocab = set([])
    dataset = []
    with open(file_name) as f:
        for line in f.readlines():
            example = line.split()
            example[0] = ["IN=" + x for x in example[0].split(',')]
            example[2] = ["OUT=" + x for x in example[2].split(',')]
            dataset.append(example)
            char_vocab.update(example[1].lower())
            char_vocab.update(example[3].lower())
            tag_vocab.update(example[0])
            tag_vocab.update(example[2])
    char_vocab.update(["<unk>", "<pad>"])
    tag_vocab.add("<unk>")

    return np.array(dataset), sorted(char_vocab), sorted(tag_vocab)

def preprocess_data(dataset, char_vocab, tag_vocab, Tx, Ty):
    input_vocab = concat_vocabs(char_vocab, tag_vocab)
    input_vocab_dict = {val:ind for ind, val in enumerate(input_vocab)}
    output_vocab_dict = {val:ind for ind, val in enumerate(char_vocab)}

    X = [x[0] + x[2] + list(x[1]) for x in dataset]
    X = [get_indices(x, input_vocab_dict, Tx) for x in X]
    Y = [get_indices(y, output_vocab_dict, Ty) for y in dataset[:,3]]
    print(dataset[0,3])
    print(Y[0])

    return np.array(X), np.array(Y), input_vocab, char_vocab


def get_indices(input, vocab, length):
    if (len(input)) > length:
        x = input[:length]
    indices = [vocab.get(x, "<unk>") for x in input]
    while len(indices) < length:
        indices.append(vocab.get("<pad>"))
    return indices

def concat_vocabs(vocab1, vocab2):
    s = set(vocab1)
    concat = vocab1.copy()
    for entry in vocab2:
        if entry in s:
            continue
        concat.append(entry)
        s.add(entry)
    return concat
