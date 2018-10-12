import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_data(inputfn, feature_size):
    x = []
    y = []
    with open(inputfn, 'r') as f:
        for line in f.readlines():
            row = line.strip('\n').split('\t')
            label = int(row[0])
            feature = np.zeros(feature_size+1)
            feature[0] = 1
            for word in row[1:]:
                index = int(word.split(':')[0])
                feature[index+1] = 1
            x.append(feature)
            y.append(label)
    return np.array(x), np.array(y)


def logistic_regression_step(x, y, theta):
    return x * (y - np.exp(np.dot(theta, x)) / (1 + np.exp(np.dot(theta, x))))


def logistic_regression(x, y, theta, lr=0.1):
    n = x.shape[0]
    for i in range(n):
        theta += lr * logistic_regression_step(x[i], y[i], theta)
    return theta


def get_labels(theta, x):
    p = 1. / (1. + np.exp(-np.dot(x, theta)))
    p[p<0.5] = 0
    p[p>=0.5] = 1
    return p.astype(int)


def write_labels(theta, x, outputfn):
    with open(outputfn, 'w') as f:
        labels = get_labels(theta, x)
        for label in labels:
            f.write('{}\n'.format(label))
    return labels


def get_error(theta, pred, y):
    error = 1. - np.mean(y == pred)
    return error


def get_negative_loglikelihood(theta, x, y):
    p = 1. / (1. + np.exp(-np.dot(x, theta)))
    p[y==0] = 1 - p[y==0]
    return np.mean(-np.log(p)) 


def write_metrics(theta, train_p, train_y, test_p, test_y, outputfn):
    with open(outputfn, 'w') as f:
        f.write('error(train): {:.6f}\n'.format(get_error(theta, train_p, train_y)))
        f.write('error(test): {:.6f}\n'.format(get_error(theta, test_p, test_y)))


def main(args):
    with open(args.dict_input, 'r') as f:
        feature_size = len(f.readlines())
    train_x, train_y = load_data(args.train_input, feature_size)
    valid_x, valid_y = load_data(args.valid_input, feature_size)
    test_x, test_y = load_data(args.test_input, feature_size)

    valid_likelihood = []
    train_likelihood = []
    # training
    theta = np.zeros(feature_size + 1)
    for i in range(args.num_epoch):
        theta = logistic_regression(train_x, train_y, theta)
        valid_error = get_error(theta, get_labels(theta, valid_x), valid_y)
        print('epoch {}: validation error = {}'.format(i, valid_error))

        valid_likelihood.append(get_negative_loglikelihood(theta, valid_x, valid_y))
        train_likelihood.append(get_negative_loglikelihood(theta, train_x, train_y))
        print(valid_likelihood[-1], train_likelihood[-1])

    train_p = write_labels(theta, train_x, args.train_out)
    test_p = write_labels(theta, test_x, args.test_out)
    write_metrics(theta, train_p, train_y, test_p, test_y, args.metrics_out)
    np.save('train.npy', np.array(train_likelihood))
    np.save('valid.npy', np.array(valid_likelihood))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train_input', help='path to the formatted training input .tsv file')
    parser.add_argument('valid_input', help='path to the formatted validation input .tsv file ')
    parser.add_argument('test_input', help='path to the formatted test input .tsv file')
    parser.add_argument('dict_input', help='path to the dictionary input .txt file')
    parser.add_argument('train_out', help='path to output .tsv file to which the prediction on the training data should be written')
    parser.add_argument('test_out', help='path to output .tsv file to which the prediction on the test data should be written')
    parser.add_argument('metrics_out', help='path of the output .txt file to which metrics such as train and test error should be writte')
    parser.add_argument('num_epoch', type=int, help='integer specifying the number of times SGD loops through all of the training data')

    args = parser.parse_args()
    main(args)