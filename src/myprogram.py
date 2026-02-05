#!/usr/bin/env python
# pickle is used to save and load python objects
import os, pickle
#Counter is a dictionary and default dict creates an empty counter
from collections import defaultdict, Counter
import unicodedata
#regex installed by Docker
import regex 
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
# Defines a function that takes a string and returns a list of tokens (grapheme clusters)
# also converts to NFC form
def normalize_and_graphemes(text: str):
    text = unicodedata.normalize("NFC", text)
    return regex.findall(r"\X", text)

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    
    # constuctor, building a 1-gram through 6-gram stats 
    def __init__(self, max_n: int = 6):
        self.max_n = max_n
        # counts[n][context_tuple] -> Counter(next_token)
        self.counts = {n: defaultdict(Counter) for n in range(1, max_n + 1)}
        self.top_unigrams = [" ", "e", "a"]  # fallback placeholder if there's no info yet/context doesn't render


    @staticmethod
    # This is where we'll load wikipedia data: 
        # Read wiki pages
        # normalize Unicode (NFC) (split by letters not words)
        # split into grapheme clusters
        # yeild text to "run_train"
    def load_training_data(path: str):
        # Currently: assumes we have a local text file of Wikipedia lines/docs.
        # we should/can change this to read many files, jsonl, etc.

        # TO DO PATH
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip("\n")
                if line:
                    yield line
    
    # Helper Method
    def update_counts_from_tokens(self, toks):
        # Looping through every token/grapheme in the doc
        for t in toks:
            self.counts[1][()][t] += 1

        # n-grams for n>=2
        for n in range(2, self.max_n + 1):
            k = n - 1 # context length
            if len(toks) <= k: # how many previous tokens we condition on, if doc is too short skip w/continue
                continue
            for i in range(k, len(toks)): # we go through window of tokens
                # predict tokens[i] from previous k tokens
                context = tuple(toks[i-k:i])
                nxt = toks[i]
                # increase count based on found context
                self.counts[n][context][nxt] += 1
                # the above part is collecting the statistics 


    # Helper Method
    def finalize(self, top_k_unigrams=3):
        # retrieve the unigram counts
        uni = self.counts[1][()]
        # if data was seen return list sorted by counts
        if len(uni) > 0:
            self.top_unigrams = [t for t, _ in uni.most_common(top_k_unigrams)]
        else:
            # Otherwise using the same placeholder fallback for now
            self.top_unigrams = [" ", "e", "a"]


    @classmethod
    # This is where we'll Test:
        # Read test file line-by line
        # remove trailing newlines
        # each line is treated as the text typed so dar
        #returns a list of strings
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname, encoding="utf-8") as f:

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
    def run_train(self, work_dir: str):
        # self.max_n = 6

        train_path = os.path.join(work_dir, "wiki.txt")
        
        max_lines = 200000 # training cap so we don't train infinitely 
        seen = 0 # counter

        # streaming docs in
        for doc in MyModel.load_training_data(train_path):
            # convert each doc with helper method to grapheme tokens & NFC normalization
            toks = normalize_and_graphemes(doc)
            self.update_counts_from_tokens(toks)
            seen += 1
            if seen >= max_lines:
                break

        self.finalize()
        self.save(work_dir)
    

        
    # Backoff Smoothing:
    # using a higher n when possible a lower n if needed 
    
    def predict_top3(self, context_text: str):
        toks = normalize_and_graphemes(context_text)

        for n in range(self.max_n, 1, -1): # putting input context into our beloved graphemes
            k = n - 1
            if len(toks) < k: # if fewer than k, still can't form the context
                continue
            ctx = tuple(toks[-k:])
            counter = self.counts[n].get(ctx)
            if counter and len(counter) > 0:
                # this means we've seen this context before in training
                preds = [t for t, _ in counter.most_common(3)]
                if len(preds) < 3:
                    preds += self.top_unigrams[: (3 - len(preds))]
                return preds[:3]

        # gets top-3 predicted graphemes and pads it if not enough was given.
        preds = self.top_unigrams[:3]
        if len(preds) < 3:
            preds += [" ", "e", "a"][: (3 - len(preds))]
        return preds[:3]
    
    # We need this to: 
        # normalize inp (NFC --> getting identical letters to look the same in unicode)
        # return a list of strings where each string is the 3 predicted characters 
        # normalize to NFC, extract context (n-1) grapheme, the score next grapheme candidates 
        # then applying smoothing/backoff, then return exactly 3 graphemes
    def run_pred(self, test_data: list[str]) -> list[str]:
        preds = []
        for context in test_data:
            top3 = self.predict_top3(context)
            preds.append("".join(top3))
        return preds
        
    # We serialize trained data so that run_pred can do lookups
    def save(self, work_dir):
        # Ensures directory exists + retrieves it
        os.makedirs(work_dir, exist_ok=True)
        path = os.path.join(work_dir, "model.checkpoint")

        with open(path, "wb") as f:
            # does the serialization containing the model states
            pickle.dump(
                {"max_n": self.max_n, "counts": self.counts, "top_unigrams": self.top_unigrams},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    # saves what we currently have in memory to be used again next time
    def load(cls, work_dir):
        path = os.path.join(work_dir, "model.checkpoint")
        with open(path, "rb") as f: #rb for read-binary
            obj = pickle.load(f)  # deserializing the dictionary saved 
        m = MyModel(max_n=obj["max_n"]) # recreating the model with the same max_n
        m.counts = obj["counts"] # restoring 
        m.top_unigrams = obj["top_unigrams"]
        return m


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
        # train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(args.work_dir)
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
