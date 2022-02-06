#!/usr/bin/env python
# utils.py
# Author: Julie Kallini

import pandas as pd
from multipledispatch import dispatch
import wordnet_relations as wr
from upos import upos

# Conjunctions under analysis
CONJUNCTIONS = ['and', 'or', 'but', 'nor']


@dispatch(str, str)
def pretty_print(input_file, output_file):
    '''
    Convert a CSV file of coordination phrases into a pretty-printed RTF file.

    @param input_file (str): path to the input CSV file.
    @param output_file (str): path to the output RTF file.
    '''

    input_df = pd.read_csv(input_file, index_col=None, header=0)
    pretty_print(input_df, output_file)


@dispatch(pd.DataFrame, str)
def pretty_print(input_df, output_file):
    '''
    Convert a DataFrame of coordination phrases into a pretty-printed RTF file.

    @param input_df (DataFrame): input coordination phrases
    @param output_file (str): path to the output RTF file
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

        label = cat1 + "+" + cat2

        if index == 0:
            out_file.write('\\b ' + label + '\\b0\line\line\n')
            section = label
        elif section != label:
            out_file.write('\line\line\\b ' + label + '\\b0\line\line\n')
            section = label

        conj1_labeled = '\\b ' + '[' + cat1 + ' ' + conj1 + ']' + '\\b0 '
        conj2_labeled = '\\b ' + '[' + cat2 + ' ' + conj2 + ']' + '\\b0 '

        # This is imperfect since the same word can appear
        # multiple times in the sentence.
        sent = sent.replace(conj1, conj1_labeled)
        sent = sent.replace(conj2, conj2_labeled)

        out_file.write(str(index + 1) + '. ' + sent + '\line\n')

    out_file.write("}")
    out_file.close()


def likes_by_category(df):
    '''
    Split a DataFrame of coordinations into four DataFrames containing
    like coordinations for each of the four open-class syntactic categories:
    nouns, verbs, adjectives, and adverbs.

    @param df (DataFrame): coordination phrases
    @return (4-tuple of DataFrames): (nouns, verbs, adjps, advps)
    '''

    nouns = df[(df['1st Conjunct Category'] == upos.NOUN) & (
        df['2nd Conjunct Category'] == upos.NOUN)]

    verbs = df[(df['1st Conjunct Category'] == upos.VERB) & (
        df['2nd Conjunct Category'] == upos.VERB)]

    adjps = df[(df['1st Conjunct Category'] == upos.ADJ) & (
        df['2nd Conjunct Category'] == upos.ADJ)]

    advps = df[(df['1st Conjunct Category'] == upos.ADV) & (
        df['2nd Conjunct Category'] == upos.ADV)]

    return (nouns, verbs, adjps, advps)


def likes_df(df):
    '''
    Returns a DataFrame of the like coordinations contained in the
    given DataFrame. Like coordinations must have conjuncts that are
    members of one of the four open-class syntactic categories:
    nouns, verbs, adjectives, and adverbs.

    @param df (DataFrame): coordination phrases
    @return (DataFrame): like coordination phrases
    '''

    likes = pd.concat(list(likes_by_category(df)),
                      axis=0, ignore_index=True)

    return likes


def unlikes_df(df):
    '''
    Returns a DataFrame of the unlike coordinations contained in the
    given DataFrame.

    @param df (DataFrame): coordination phrases
    @return (DataFrame): unlike coordination phrases
    '''
    all_categories = [upos.NOUN, upos.VERB, upos.ADJ, upos.ADV]

    df = df[df['1st Conjunct Category'].isin(all_categories)]
    df = df[df['2nd Conjunct Category'].isin(all_categories)]

    # Get unlike category combinations
    unlikes = df.loc[df['1st Conjunct Category']
                     != df['2nd Conjunct Category']]

    return unlikes


def analyze_synonymy(df):
    """
    Run synonymy analysis on all categories in the given DataFrame.
    Returns DataFrame with new boolean 'Synonyms?' column.

    @param df (DataFrame): coordination phrases
    @return (DataFrame): coordinations with synonymy relation column
    """

    df = df.copy()

    # Ensure Dataframe only contains conjuncts of like categories
    df = likes_df(df)

    df['Synonyms?'] = df.apply(lambda row: wr.synonyms(
        str(row['1st Conjunct Lemma']),
        str(row['2nd Conjunct Lemma']),
        str(row['1st Conjunct Category'])), axis=1)

    return df[df['Synonyms?'].notnull()]


def analyze_antonymy(df):
    """
    Run antonymy analysis on adjective and adverb-like categories
    in the given DataFrame. Returns DataFrame with a new 'Antonyms?'
    column.

    @param df (DataFrame): coordination phrases
    @return (DataFrame): coordinations with antonymy relation column
    """

    df = df.copy()

    # Ensure DataFrame only contains adjective/adverb categories
    df = df[((df['1st Conjunct Category'] == upos.ADJ) & (df['2nd Conjunct Category'] == upos.ADJ)) |
            ((df['1st Conjunct Category'] == upos.ADV) & (df['2nd Conjunct Category'] == upos.ADV))]

    df['Antonyms?'] = df.apply(lambda row: wr.antonyms(
        str(row['1st Conjunct Lemma']),
        str(row['2nd Conjunct Lemma']),
        str(row['1st Conjunct Category'])), axis=1)

    return df[df['Antonyms?'].notnull()]


def analyze_hypernymy(df):
    """
    Run hypernymy analysis on noun-like and verb-like categories in the given
    DataFrame. Returns DataFrame with new columns '1st Conjunct Hypernym?'
    and '2nd Conjunct Hypernym?'.

    @param df (DataFrame): coordination phrases
    @return (Dataframe): coordinations with hypernymy relation columns
    """

    df = df.copy()

    # Ensure DataFrame only contains nominal/verbal categories
    df = df[((df['1st Conjunct Category'] == upos.NOUN) & (df['2nd Conjunct Category'] == upos.NOUN)) |
            ((df['1st Conjunct Category'] == upos.VERB) & (df['2nd Conjunct Category'] == upos.VERB))]

    df['1st Conjunct Hypernym?'] = df.apply(lambda row: wr.is_hypernym(
        str(row['1st Conjunct Lemma']),
        str(row['2nd Conjunct Lemma']),
        str(row['1st Conjunct Category'])), axis=1)
    df['2nd Conjunct Hypernym?'] = df.apply(lambda row: wr.is_hypernym(
        str(row['2nd Conjunct Lemma']),
        str(row['1st Conjunct Lemma']),
        str(row['1st Conjunct Category'])), axis=1)

    df = df[df['1st Conjunct Hypernym?'].notnull()]
    df = df[df['2nd Conjunct Hypernym?'].notnull()]

    return df


def analyze_cohyponymy(df):
    """
    Run co-hyponymy analysis on noun-like and verb-like categories in the given
    DataFrame. Returns DataFrame with new column 'Co-hyponyms?'.

    @param df (DataFrame): coordination phrases
    @return (Dataframe): coordinations with co-hyponymy relation column
    """

    df = df.copy()

    # Ensure DataFrame only contains nominal/verbal categories
    df = df[((df['1st Conjunct Category'] == upos.NOUN) & (df['2nd Conjunct Category'] == upos.NOUN)) |
            ((df['1st Conjunct Category'] == upos.VERB) & (df['2nd Conjunct Category'] == upos.VERB))]

    df['Co-hyponyms?'] = df.apply(lambda row: wr.co_hyponyms(
        str(row['2nd Conjunct Lemma']),
        str(row['1st Conjunct Lemma']),
        str(row['1st Conjunct Category'])), axis=1)

    return df[df['Co-hyponyms?'].notnull()]


def analyze_entailment(df):
    """
    Run entailment analysis on verb-like categories in the given DataFrame.
    Returns DataFrame with new columns '1st Conjunct Entails 2nd?' and
    '2nd Conjunct Entails 1st?'.

    @param df (DataFrame): coordination phrases
    @return (Dataframe): coordinations with entailment relation columns
    """

    df = df.copy()

    # Ensure conjuncts have verbal categories
    df = df[(df['1st Conjunct Category'] == upos.VERB) &
            (df['2nd Conjunct Category'] == upos.VERB)]

    df['1st Conjunct Entails 2nd?'] = df.apply(lambda row: wr.entails(
        str(row['1st Conjunct Lemma']),
        str(row['2nd Conjunct Lemma']),
        str(row['1st Conjunct Category'])), axis=1)
    df['2nd Conjunct Entails 1st?'] = df.apply(lambda row: wr.entails(
        str(row['2nd Conjunct Lemma']),
        str(row['1st Conjunct Lemma']),
        str(row['1st Conjunct Category'])), axis=1)

    df = df[df['1st Conjunct Entails 2nd?'].notnull()]
    df = df[df['2nd Conjunct Entails 1st?'].notnull()]

    return df
