#!/usr/bin/env python
# PTB.py
# Author: Julie Kallini

import pandas as pd
import os
import re
from nltk import ParentedTree
from tqdm import tqdm
import argparse


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

    for s in tree.subtrees(
            lambda t: t.label() == "CC" and
            (len(list(t.parent())) == 3 or len(list(t.parent())) == 4)):

        parent = s.parent()
        phrase_cat = parent.label()
        phrase_text = get_tree_text(parent)

        # Simple three-prong coordination phrases
        if len(list(parent)) == 3:
            # Get left ad right siblings
            left = s.left_sibling()
            right = s.right_sibling()

            if left is None or right is None:
                continue

            conjunct1 = (left.label(), get_tree_text(left))
            conjunct2 = (right.label(), get_tree_text(right))
            conjunction = get_tree_text(s)
            phrases.append(
                ([conjunct1, conjunct2], conjunction, phrase_cat, phrase_text))

        # "neither-nor" coordination phrases
        elif get_tree_text(parent[0]) == 'neither' and get_tree_text(parent[2]) == 'nor':
            left = parent[1]
            right = parent[3]
            conjunct1 = (left.label(), get_tree_text(left))
            conjunct2 = (right.label(), get_tree_text(right))
            conjunction = 'nor'
            phrases.append(
                ([conjunct1, conjunct2], conjunction, phrase_cat, phrase_text))

        # VPs with both conjuncts as complements
        elif parent.label() == 'VP' and parent[2].label() == 'CC':
            left = parent[1]
            right = parent[3]
            conjunct1 = (left.label(), get_tree_text(left))
            conjunct2 = (right.label(), get_tree_text(right))
            conjunction = get_tree_text(parent[2])
            phrases.append(
                ([conjunct1, conjunct2], conjunction, phrase_cat, phrase_text))

    return phrases


def get_coordphrases_ptbext(tree):
    '''
    Function: Find all coordination phrases of the given NLTK tree,
    assuming the format is from the PTB extension.

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


def get_args():
    '''
    Parse command-line arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Extract coordinations from PTB style input files.')
    parser.add_argument('--gum', dest='gum', action='store_true',
                        help='Indicates that the input files are in GUM format')
    parser.add_argument('input_files', nargs='+', type=str,
                        help='path to input PTB file(s)')
    return parser.parse_args()


if __name__ == "__main__":
    '''
    Main function.
    '''
    args = get_args()

    i = 1
    tot = str(len(args.input_files))

    for file in args.input_files:

        print("(" + str(i) + "/" + tot + ")")
        print("Parsing coordinations from {}...".format(file))

        f = open(file, 'r', encoding='utf-8')
        lines = f.readlines()

        data = []

        for sent_tree in tqdm(lines):

            # Parse this sent_tree into an NLTK tree object
            tree = ParentedTree.fromstring(sent_tree)

            # Get all phrases in this tree
            if args.gum:
                coordphrases = get_coordphrases(tree)
            else:
                coordphrases = get_coordphrases_ptbext(tree)

            for phrase in coordphrases:

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

        dest_name = file.split('.')[0]

        df.to_csv(dest_name + '.csv', index=False)

        print("All done! The result is stored in {}.csv.".format(dest_name))

        i += 1
