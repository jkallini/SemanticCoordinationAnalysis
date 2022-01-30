#!/usr/bin/env python
# PTB.py
# Author: Julie Kallini

import pandas as pd
import os
import re
from nltk import ParentedTree
from tqdm import tqdm
from utils import rawgencount


def get_tree_text(tree):
    '''
    Function: Get the sentence text of the given NLTK tree. Removes all
    trace leaves, or leaves with shape: *.*.

    @param tree (Tree): NLTK tree object
    @return (str): sentence text
    '''

    non_trace_leaves = []
    for leaf in tree.leaves():
        if not re.search('\*', leaf):
            non_trace_leaves.append(leaf)

    return " ".join(non_trace_leaves)


def get_coordphrases(tree):
    '''
    Function: Find all coordination phrases of the given NLTK tree.

    @param tree (Tree): NLTK tree object
    @return (list of tuples): phrase
        a phrase is a 4 tuple containing a list of conjuncts, a conjunction
        (string), a phrase category (string), and a phrase text (string)
        a conjunct is a tuple containing its category (string) and text (string)
    '''
    phrases = []

    # In PTB.ext, all coordination phrases are annotated with "-CCP"
    for s in tree.subtrees(lambda t: "-CCP" in t.label()):
        conjuncts = []
        conjunction = None
        for child in s:

            # Find all conjuncts (labeled with "-COORD")
            if "-COORD" in child.label():
                conjuncts.append((child.label(), get_tree_text(child)))

            # Find conjunction (labeled with "CC-CC")
            if "CC-CC" in child.label():
                conjunction = get_tree_text(child)

        if conjunction is not None:
            phrase_cat = s.label()
            phrase_text = get_tree_text(s)
            phrases.append((conjuncts, conjunction, phrase_cat, phrase_text))

    return phrases


if __name__ == "__main__":
    '''
    Main function.
    '''
    print("Beginning parse of PTB.ext...")

    data = []
    num_lines = rawgencount("PTB.ext")

    with open('PTB.ext', encoding='utf-8') as f:
        for sent_tree in tqdm(f, total=num_lines):

            # Parse this sent_tree into an NLTK tree object
            tree = ParentedTree.fromstring(sent_tree)

            # Get all phrases in this tree
            for phrase in get_coordphrases(tree):

                conjuncts = phrase[0]
                conjunction = phrase[1]
                phrase_cat = phrase[2]
                phrase_text = phrase[3]
                sent_text = get_tree_text(tree)

                # Only include two-termed coordinations
                if len(conjuncts) != 2:
                    continue

                row = []
                for (cat, text) in conjuncts:
                    row.append(cat.split('-')[0])
                    row.append(text)

                row.append(phrase_cat.split('-')[0])
                row.append(phrase_text)
                row.append(conjunction)
                row.append(sent_text)
                row.append(sent_tree)

                data.append(row)

    columns = ['1st Conjunct Category', '1st Conjunct Text',
               '2nd Conjunct Category', '2nd Conjunct Text',
               'Phrase Category', 'Phrase Text',
               'Conjunction', 'Sentence Text', 'Sentence Parse Tree']

    df = pd.DataFrame(data, columns=columns)

    if not os.path.exists('csv/PTB/'):
        os.makedirs('csv/PTB/')

    df.to_csv('csv/PTB/PTB_ccps.csv', index=False)

    print("All done! The result is stored in csv/PTB/PTB.csv.")
