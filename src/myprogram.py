#!/usr/bin/env python
import os
import string
import random # currently because of placeholder logic 
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

"""
    Notes: 
        Data loading: load_training_data & load_test_data
        Model behavior: run_train & run_pred
        Persistence: save & load
        Output Formatting: write_pred
"""

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    @classmethod
    # This is where we'll load wikipedia data: 
        # Read wiki pages
        # normalize Unicode (NFC) (split by letters not words)
        # split into grapheme clusters
        # yeild text to "run_train"
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
        return []

    @classmethod
    # This is where we'll Test:
        # Read test file line-by line
        # remove trailing newlines
        # each line is treated as the text typed so dar
        #returns a list of strings
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    # Writes predictions to a file
        # each prediction is a string of length 3? 
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    # This is where we learn 
        # Build 5-8 grams over grapheme clusters
        # apply normalization 
        # stores counts for inference
    def run_train(self, data, work_dir):
        # your code here
        pass
    
    # We need this to: 
        # normalize inp (NFC --> getting identical letters to look the same in unicode)
        # return a list of strings where each string is the 3 predicted characters 
        # normalize to NFC, extract context (n-1) grapheme, the score next grapheme candidates 
        # then applying smoothing/backoff, then return exactly 3 graphemes

    def run_pred(self, data):
        # your code here
        preds = []
        all_chars = string.ascii_letters
        for inp in data:
            # this model just predicts a random character each time currently 
            top_guesses = [random.choice(all_chars) for _ in range(3)]
            preds.append(''.join(top_guesses))
        return preds

    # searlizes trained data so that run_pred can do lookups
    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    # saves what we currently have in memory to be used again next time
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            dummy_save = f.read()
        return MyModel()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
