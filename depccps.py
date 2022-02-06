#!/usr/bin/env python
# depccps.py
# Author: Julie Kallini


from conllu import parse_incr
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import argparse


def get_conjuncts_from_ids(ids, tokenlist):
    '''
    Get (lemma, pos) tuples from the list of integer ids representing
    words within the given tokenlist. Tuples are ordered by conjunct
    appearance in the the sentence.

    @param ids (list of ints): list of integer ids
    @param tokenlist (TokenList): dependency parse of a CoNLL-U sentence
    @return (list of (str, str) tuples): list of lemma, pos) tuples
                                         representing conjuncts
    '''

    # Get tokens of words corresponding to these ids
    conjunct_tokens = tokenlist.filter(id=lambda x: x in ids)

    # Get (lemma, pos) tuples for these tokens
    conjuncts = []
    for token in conjunct_tokens:
        conjuncts.append((token['lemma'], token['upos']))

    return conjuncts


def get_coordphrases(tokenlist, cc='and'):
    '''
    Get coordination phrases in the given tokenlist that use the given
    coordinating conjunction cc.

    @param tokenlist (TokenList): dependency parse of a CoNLL-U sentence
    @param cc (str): coordinating conjunction
    @return (list of (str, list) tuples): list of (conjunction, conjunct list)
            tuples representing coordination phrases; a conjunct is
            a (lemma, pos) tuple of strings
    '''

    # Initialize dictionary mapping first conjuncts to a list of conjuncts
    # that follow it in a coordination phrase
    conjunct_id_sets = defaultdict(set)

    # Find all tokens with a 'conj' dependency
    for tok in tokenlist.filter(deprel='conj'):

        # Get first conjunct (i.e., the head of the 'conj' dependency)
        # if the coordination uses the given conjunction cc
        first_conjunct = [id for (dep, id) in tok['deps'] if dep == 'conj:'+cc]

        # There is either one first conjunct or none
        assert(len(first_conjunct) <= 1)
        if len(first_conjunct) != 1:
            continue
        first_conjunct_id = first_conjunct[0]

        # Add both conjuncts to the id set
        conjunct_id_sets[first_conjunct_id].add(first_conjunct_id)
        conjunct_id_sets[first_conjunct_id].add(tok['id'])

    # Return coordphrases as list of (conjunction, conjunct list) tuples
    coordphrases = [(cc, get_conjuncts_from_ids(list(s), tokenlist))
                    for _, s in conjunct_id_sets.items()]

    return coordphrases


def get_args():
    '''
    Parse command-line arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Preprocess raw CoNLL-U file(s).')
    parser.add_argument('input_files', nargs='+', type=str,
                        help='path to input CoNLL-U file(s)')
    return parser.parse_args()


if __name__ == "__main__":
    '''
    Main function.
    '''

    args = get_args()

    i = 1
    tot = str(len(args.input_files))

    # Iterate over input files
    for file_name in tqdm(args.input_files):

        file = open(file_name, "r", encoding="utf-8")
        data = []

        # Get tokenlist for each sentence in file
        for tokenlist in parse_incr(file):

            sent_text = tokenlist.metadata['text']

            # Get coordination phrases for each coordinating conjunction
            and_coordphrases = get_coordphrases(tokenlist, cc='and')
            or_coordphrases = get_coordphrases(tokenlist, cc='or')
            but_coordphrases = get_coordphrases(tokenlist, cc='but')
            nor_coordphrases = get_coordphrases(tokenlist, cc='nor')
            coordphrases = [*and_coordphrases, *or_coordphrases,
                            *but_coordphrases, *nor_coordphrases]

            # Iterate over coordination phrases
            for cc, conjuncts in coordphrases:

                # Only include two-termed coordinations
                if len(conjuncts) != 2:
                    continue

                row = []

                # Add conjuncts' categories and texts
                for (text, cat) in conjuncts:
                    row.append(cat)
                    row.append(text)

                # Add conjunction and sentence text
                row.append(cc)
                row.append(sent_text)

                data.append(row)

        columns = ['1st Conjunct Category', '1st Conjunct Text',
                   '2nd Conjunct Category', '2nd Conjunct Text',
                   'Conjunction', 'Sentence Text']

        # Create DataFrame and write it to file
        df = pd.DataFrame(data, columns=columns)
        dest_name = file_name.split('.')[0]
        df.to_csv(dest_name + '.csv', index=False)

        i += 1
