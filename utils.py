import numpy as np
from keras.utils import to_categorical


def load_dataset(file_name):
    dataset = []
    with open(file_name) as f:
        for line in f.readlines():
            example = line.split()
            example[1] = example[1].lower()
            example[3] = example[3].lower()
            example[0] = example[0].split(",")
            example[2] = example[2].split(",")
            dataset.append(example)
    return np.array(dataset)


def enhance_dataset(dataset):
    dataset_cpy = dataset.copy()
    dataset_cpy[:, [1, 3]] = dataset_cpy[:, [3, 1]]
    dataset_cpy[:, [0, 2]] = dataset_cpy[:, [2, 0]]
    dataset = np.concatenate((dataset, dataset_cpy), axis=0)
    return dataset


def preprocess_data(dataset):
    dataset = enhance_dataset(dataset)
    dataset = edit_tags(dataset)
    char_vocab, tag_vocab = get_vocabs(dataset)
    input_vocab = concat_vocabs(char_vocab, tag_vocab)
    input_vocab_dict = {val:ind for ind, val in enumerate(input_vocab)}
    output_vocab_dict = {val:ind for ind, val in enumerate(char_vocab)}
    Tx, Ty = get_seq_lengths(dataset)

    X = [np.concatenate((x[0], x[2], list(x[1]))) for x in dataset]
    X = np.array([get_indices(x, input_vocab_dict, Tx) for x in X])
    Y = np.array([get_indices(y, output_vocab_dict, Ty) for y in dataset[:, 3]])
    Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(input_vocab)), X)))
    Yoh = np.array(list(map(lambda y: to_categorical(y, num_classes=len(char_vocab)), Y)))

    return X, Y, Xoh, Yoh, input_vocab, char_vocab, Tx, Ty


def edit_tags(dataset):
    for i in range(0, dataset.shape[0]):
        dataset[i, 0] = np.array(["IN=" + x for x in dataset[i, 0]])
        dataset[i, 2] = np.array(["OUT=" + x for x in dataset[i, 2]])
    return dataset


def get_vocabs(dataset):
    char_vocab = set([])
    tag_vocab = set([])
    for i in range(0, dataset.shape[0]):
        char_vocab.update(dataset[i, 1].lower())
        char_vocab.update(dataset[i, 3].lower())
        tag_vocab.update(dataset[i, 0])
        tag_vocab.update(dataset[i, 2])
    char_vocab.update(["<pad>", "<unk>"])
    tag_vocab.add("<unk>")
    return sorted(char_vocab), sorted(tag_vocab)


def get_indices(input, vocab, length):
    if (len(input)) > length:
        x = input[:length]
    indices = [vocab.get(x, vocab["<unk>"]) for x in input]
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


def get_seq_lengths(dataset):
    max_word_len = max([len(x) for x in np.append(dataset[:,1], dataset[:,3])])
    max_tag_len = max([len(x) for x in np.append(dataset[:,0], dataset[:,2])])
    return 2*max_tag_len + max_word_len, max_word_len

