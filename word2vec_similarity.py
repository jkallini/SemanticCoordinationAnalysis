#!/usr/bin/env python
# word2vec_similarity.py
# Author: Julie Kallini
# Utilizes Google's pre-trained word embeddings: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

import gensim
import argparse
import pandas as pd


def get_args():
    '''
    Parse command-line arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Add Word2Vec similarity stats to coordinations in the csv input file(s).')
    parser.add_argument('input_files', nargs='+', type=str,
                        help='path to input csv file(s)')
    return parser.parse_args()


def similarity(model, word1, word2):
    '''
    Get the model similarity of word1 and word2, if they are in the vocabulary.

    @param model: gensim Word2Vec model
    @param word1 (str): English word
    @param word2 (str): English word
    @return (float): similarity of word1 and word2
    '''
    try:
        return model.similarity(word1, word2)
    except:
        return None


if __name__ == "__main__":
    '''
    Main function.
    '''

    args = get_args()

    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.KeyedVectors.load_word2vec_format(
        './GoogleNews-vectors-negative300.bin', binary=True)

    i = 1
    tot = str(len(args.input_files))
    for file in args.input_files:

        print("(" + str(i) + "/" + tot + ")")

        print("Getting Word2Vec similarity of conjuncts in " + file + "...")

        df = pd.read_csv(file)

        df['Cosine Similarity'] = df.apply(lambda row: similarity(
            model,
            str(row['1st Conjunct Lemma']),
            str(row['2nd Conjunct Lemma']),
        ), axis=1)

        df.to_csv(file, index=False)

        print("Similarity analysis done! Result stored in " + file + ".")

        i = i + 1
