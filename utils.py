import pandas as pd

def pretty_print(input_file, output_file):
    '''
    Convert a CSV file of coordination phrases into a pretty-printed RTF file.

    Keyword Arguments:
      input_file -- path to the input CSV file.
      output_file -- path to the output RTF file.
    Return:
      None
    '''

    in_file = pd.read_csv(input_file, index_col=None, header=0)
    out_file = open(output_file,'w')
    out_file.write("{\\rtf1\n")

    section = ''
    for index, row in in_file.iterrows():
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

