#!/usr/bin/env python
# GUM.py
# Author: Julie Kallini

import os
import argparse


def preprocess_GUM(lines):
    '''
    Processes lines of GUM files into format suitable to be
    parsed as NLTK trees.

    @param lines (list of str): lines of GUM file
    @return (list of str): GUM file represented as one tree per line
    '''

    # Trees are on multiple lines separated by two newlines in GUM files
    # We split the data based on the appearance of two newlines
    lines = "".join(lines).split(os.linesep + os.linesep)

    # Normalize all other whitespace
    lines = [' '.join(l.split()) + os.linesep for l in lines]

    return lines


def get_args():
    '''
    Parse command-line arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Preprocess raw GUM files.')
    parser.add_argument('input_files', nargs='+', type=str,
                        help='path to input GUM file(s)')
    return parser.parse_args()


if __name__ == "__main__":
    '''
    Main function.
    '''

    args = get_args()

    i = 1
    tot = str(len(args.input_files))

    all_lines = []

    print("Preprocessing GUM files...")

    for file_name in args.input_files:
        file = open(file_name, 'r')
        lines = preprocess_GUM(file.readlines())
        all_lines = all_lines + lines

    # Write lines to outfile
    outfile = open('csv/GUM/GUM.ptb', 'w', encoding='utf-8')
    outfile.writelines(all_lines)
    outfile.close()

    print("All done! The result is stored in csv/GUM/GUM.ptb.")
