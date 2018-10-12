import argparse
import numpy as np


def get_dict(fn):
    with open(fn, 'r') as f:
        dictionary = {}
        for line in f.readlines():
            data = line.strip('\n').split(' ')
            dictionary[data[0]] = int(data[1])
        return dictionary


def get_features(inputfn, outputfn, dictionary, trim=4):
    with open(inputfn, 'r') as f:
        with open(outputfn, 'w') as out:
            for line in f.readlines():
                row = line.strip('\n').split('\t')
                label = row[0]
                words = row[1].split(' ')
                row_dict = {}
                for word in words:
                    if word in dictionary: 
                        index = dictionary[word]
                        if index in row_dict: row_dict[index] += 1
                        else: row_dict[index] = 1
                out.write(label)
                for (key, value) in row_dict.items():
                    if value < trim or trim == 0:
                        out.write('\t{}:1'.format(key))
                out.write('\n')


def main(args):
    trim = 0 if args.feature_flag == 1 else 4
    dictionary = get_dict(args.dict_input)
    get_features(args.train_input, args.f_train_out, dictionary, trim)
    get_features(args.valid_input, args.f_valid_out, dictionary, trim)
    get_features(args.test_input, args.f_test_out, dictionary, trim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train_input', help='path to the training input .tsv file')
    parser.add_argument('valid_input', help='path to the validation input .tsv file ')
    parser.add_argument('test_input', help='path to the test input .tsv file')
    parser.add_argument('dict_input', help='path to the dictionary input .txt file')
    parser.add_argument('f_train_out', help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument('f_valid_out', help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument('f_test_out', help='path to output .tsv file to which the feature extractions on the test data should be written')
    parser.add_argument('feature_flag', type=int, help='integer taking value 1 or 2 that specifies whether to construct the Model 1 feature set or the Model 2 feature set ')

    args = parser.parse_args()
    main(args)