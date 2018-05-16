import numpy as np

PAD_TAG = "<pad>"
START_TAG = "<w>"
END_TAG = "</w>"
UNKNOWN_TAG = "<unk>"

def load_dataset(file_name):
    inputs = []
    outputs = []
    with open(file_name) as f:
        for line in f.readlines():
            example = line.split()
            example[1] = example[1].lower()
            example[3] = example[3].lower()
            example[0] = example[0].split(",")
            example[2] = example[2].split(",")
            inputs.append(example[:3])
            outputs.append(example[3])
    return np.array(inputs), np.array(list(outputs))


def enhance_dataset(inputs, outputs):
    inputs_cpy = inputs.copy()
    outputs_cpy = outputs.copy()
    inputs_cpy[:, [0, 2]] = inputs_cpy[:, [2, 0]]
    inputs_cpy[:, 1], outputs_cpy[:] = outputs_cpy[:], inputs_cpy[:, 1]
    inputs = np.concatenate((inputs, inputs_cpy), axis=0)
    outputs = np.concatenate((outputs, outputs_cpy), axis=0)
    return inputs, outputs


def preprocess_data(inputs, outputs, train=True):
    if train:
        inputs, outputs = enhance_dataset(inputs, outputs)
    inputs = edit_tags(inputs)
    inputs[:, [1, 2]] = inputs[:, [2, 1]]
    inputs = transform_to_sequences(inputs)
    input_vocab = get_vocab(inputs)
    output_vocab = get_vocab(outputs)

    return inputs, outputs, input_vocab, output_vocab
"""
    X = [np.concatenate((x[0], x[2], list(x[1]))) for x in dataset]
    X = np.array([get_indices(x, input_vocab_dict, Tx) for x in X])
    Y = np.array([get_indices(y, output_vocab_dict, Ty) for y in dataset[:, 3]])
    Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(input_vocab)), X)))
    Yoh = np.array(list(map(lambda y: to_categorical(y, num_classes=len(char_vocab)), Y)))

    return X, Y, Xoh, Yoh, input_vocab, char_vocab, Tx, Ty
"""

def edit_tags(inputs):
    for i in range(0, inputs.shape[0]):
        inputs[i, 0] = np.array(["IN=" + x for x in inputs[i, 0]])
        inputs[i, 2] = np.array(["OUT=" + x for x in inputs[i, 2]])
    return inputs


def transform_to_sequences(inputs):
    input_seq = np.array(
        [np.concatenate((inputs[i, 0], inputs[i, 1], list(inputs[i, 2])))
         for i in range(inputs.shape[0])]
    )
    return input_seq


def get_vocab(data):
    idx_to_char = {0: PAD_TAG, 1: START_TAG, 2: END_TAG, 3: UNKNOWN_TAG}
    char_to_idx = {PAD_TAG: 0, START_TAG: 1, END_TAG: 2, UNKNOWN_TAG: 3}
    char_set = set([])
    for i in range(0, data.shape[0]):
        char_set.update(data[i])
    char_set = sorted(char_set)
    for i in range(0, len(char_set)):
        idx_to_char[i+4] = char_set[i]
        char_to_idx[char_set[i]] = i+4
    return idx_to_char, char_to_idx


def get_indices(input, vocab):
    return [vocab[ch] for ch in input] + [vocab[END_TAG]]

