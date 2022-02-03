#!/usr/bin/env python
# depccps.py
# Author: Julie Kallini


from conllu import parse_incr
from collections import defaultdict
import os
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

    # Make sure the tokens are sorted
    conjunct_tokens.sort(key=lambda x: x['id'])

    # Get (lemma, pos) tuples for these tokens
    conjuncts = []
    for token in conjunct_tokens:
        conjuncts.append((token['lemma'], token['upos']))

    return conjuncts


def get_coordphrases(tokenlist, cc='and'):
    '''
    Get (lemma, pos) tuples from the list of integer ids representing
    words within the given tokenlist. Tuples are ordered by conjunct
    appearance in the the sentence.

    @param ids (list of ints): list of integer ids
    @param tokenlist (TokenList): dependency parse of a CoNLL-U sentence
    @return (list of (str, str) tuples): list of lemma, pos) tuples
                                         representing conjuncts
    '''

    conjunct_id_sets = defaultdict(set)

    # Find all
    for tok in tokenlist.filter(deprel='conj'):
        print(tok['lemma'])

        first_conjunct = [id for (dep, id) in tok['deps'] if dep == 'conj:'+cc]

        if len(first_conjunct) != 1:
          continue

        first_conjunct_id = first_conjunct[0]

        # first_conjunct_id = tok['head']

        conjunct_id_sets[first_conjunct_id].add(first_conjunct_id)
        conjunct_id_sets[first_conjunct_id].add(tok['id'])

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

    for file_name in args.input_files:

        file = open(file_name, "r", encoding="utf-8")

        data = []
        for tokenlist in tqdm(parse_incr(file)):

            sent_text = tokenlist.metadata['text']

            and_coordphrases = get_coordphrases(tokenlist, cc='and')
            or_coordphrases = get_coordphrases(tokenlist, cc='or')
            but_coordphrases = get_coordphrases(tokenlist, cc='but')
            nor_coordphrases = get_coordphrases(tokenlist, cc='nor')
            coordphrases = [*and_coordphrases, *or_coordphrases,
                            *but_coordphrases, *nor_coordphrases]

            if len(coordphrases) != 0:
              print(sent_text)
              print(coordphrases)

            for cc, conjuncts in coordphrases:
                if len(conjuncts) != 2:
                    continue

                row = []
                for (text, cat) in conjuncts:
                    row.append(cat)
                    row.append(text)

                row.append(cc)
                row.append(sent_text)

                data.append(row)

        columns = ['1st Conjunct Category', '1st Conjunct Text',
                   '2nd Conjunct Category', '2nd Conjunct Text',
                   'Conjunction', 'Sentence Text']

        df = pd.DataFrame(data, columns=columns)

        dest_name = file_name.split('.')[0]

        df.to_csv(dest_name + '.csv', index=False)

        print("All done! The result is stored in {}.csv.".format(dest_name))

        i += 1
