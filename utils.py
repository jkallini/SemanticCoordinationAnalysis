#!/usr/bin/env python
# utils.py

import pandas as pd
from multipledispatch import dispatch
import spacy

# Conjunctions under analysis
CONJUNCTIONS = ['and', 'or', 'but', 'nor']

# Categories under analysis
NOUN_CATEGORIES = ['NN', 'NNS', 'NNP', 'NNPS', 'NP', 'NX']
VERB_CATEGORIES = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VP']
ADJ_CATEGORIES = ['JJ', 'JJR', 'JJS', 'ADJP']
ADV_CATEGORIES = ['RB', 'RBR', 'RBS', 'ADVP']

PHRASAL_CATEGORIES = ['NP', 'VP', 'ADJP', 'ADVP']


@dispatch(str, str)
def pretty_print(input_file, output_file):
    '''
    Convert a CSV file of coordination phrases into a pretty-printed RTF file.

    Keyword Arguments:
      input_file -- path to the input CSV file.
      output_file -- path to the output RTF file.
    Return:
      None
    '''

    input_df = pd.read_csv(input_file, index_col=None, header=0)
    pretty_print(input_df, output_file)


@dispatch(pd.DataFrame, str)
def pretty_print(input_df, output_file):
    '''
    Convert a DataFrame of coordination phrases into a pretty-printed RTF file.

    Keyword Arguments:
      input_df -- input DataFrame of coordination phrases
      output_file -- path to the output RTF file
    Return:
      None
    '''

    out_file = open(output_file, 'w')
    out_file.write("{\\rtf1\n")

    section = ''
    for index, row in input_df.iterrows():
        sent = str(row['Sentence Text'])
        conj1 = str(row['1st Conjunct Text'])
        cat1 = str(row['1st Conjunct Category'])
        conj2 = str(row['2nd Conjunct Text'])
        cat2 = str(row['2nd Conjunct Category'])
        conjunction = str(row['Conjunction'])

        label = cat1 + "+" + cat2

        if index == 0:
            out_file.write('\\b ' + label + '\\b0\line\line\n')
            section = label
        elif section != label:
            out_file.write('\line\line\\b ' + label + '\\b0\line\line\n')
            section = label

        conj1_labeled = '\\b ' + '[' + cat1 + ' ' + conj1 + ']' + '\\b0 '
        conj2_labeled = '\\b ' + '[' + cat2 + ' ' + conj2 + ']' + '\\b0 '

        ccp = conj1 + ' ' + conjunction + ' ' + conj2
        ccp_labeled = conj1_labeled + ' ' + conjunction + ' ' + conj2_labeled
        sent = sent.replace(ccp, ccp_labeled)

        out_file.write(str(index + 1) + '. ' + sent + '\line\n')

    out_file.write("}")
    out_file.close()


def likes_df(df):
    '''
    Returns a DataFrame of the like coordinations contained in the
    given DataFrame.

    Keyword Arguments:
        df -- DataFrame containing coordinations
    Return:
        Dataframe of like coordinations
    '''

    nouns = df[(df['1st Conjunct Category'].isin(NOUN_CATEGORIES)) & (
        df['2nd Conjunct Category'].isin(NOUN_CATEGORIES))]

    verbs = df[(df['1st Conjunct Category'].isin(VERB_CATEGORIES)) & (
        df['2nd Conjunct Category'].isin(VERB_CATEGORIES))]

    adjps = df[(df['1st Conjunct Category'].isin(ADJ_CATEGORIES)) & (
        df['2nd Conjunct Category'].isin(ADJ_CATEGORIES))]

    advps = df[(df['1st Conjunct Category'].isin(ADV_CATEGORIES)) & (
        df['2nd Conjunct Category'].isin(ADV_CATEGORIES))]

    likes = pd.concat([nouns, verbs, adjps, advps],
                      axis=0, ignore_index=True)

    return likes


def unlikes_df(df):
    '''
    Returns a DataFrame of the unlike coordinations contained in the
    given DataFrame.

    Keyword Arguments:
        df -- DataFrame containing coordinations
    Return:
        Dataframe of unlike coordinations
    '''

    df = df[df['1st Conjunct Category'].isin(PHRASAL_CATEGORIES)]
    df = df[df['2nd Conjunct Category'].isin(PHRASAL_CATEGORIES)]

    # Get unlike category combinations
    unlikes = df.loc[df['1st Conjunct Category']
                     != df['2nd Conjunct Category']]

    return unlikes


def filter_conjlength(df, length):
    '''
    Returns a DataFrame of coordinations contained in the given
    DataFrame where each conjunct is at most the given length.

    Keyword Arguments:
        df -- DataFrame containing coordinations
        length -- integer length to filter coordinations
    Return:
        Dataframe of filtered coordinations
    '''
    df['Sentence Text'] = df['Sentence Text'].astype('str')
    mask1 = df['1st Conjunct Text'].str.split().str.len()
    mask2 = df['2nd Conjunct Text'].str.split().str.len()
    return df.loc[(mask1 <= length) & (mask2 <= length)]


def get_head(phrase, nlp):
    '''
    Returns the syntactic head of the phrase using spaCy's dependency
    parser, if it exists. Returns None otherwise.

    Keyword Arguments:
        phrase -- string phrase to parse
        nlp -- spaCy language model
    Return:
        Dataframe of filtered coordinations
    '''
    doc = nlp(phrase)
    sents = list(doc.sents)
    if sents != []:
        return str(list(doc.sents)[0].root)


def add_conj_heads(df):
    '''
    Adds two new columns to the given DataFrame containing the syntactic
    heads of each conjunct.

    Keyword Arguments:
        df -- DataFrame containing coordinations
    Return:
        Dataframe of coordinations with conjunct heads
    '''
    nlp = spacy.load("en_core_web_lg")
    df['1st Conjunct Head'] = df.apply(
        lambda row: get_head(str(row['1st Conjunct Text']), nlp), axis=1)
    df['2nd Conjunct Head'] = df.apply(
        lambda row: get_head(str(row['2nd Conjunct Text']), nlp), axis=1)
