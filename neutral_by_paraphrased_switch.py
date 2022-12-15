
"""
This script is used to create neutral hypothesis sentences by switching previously paraphrased sentences with the original
premise sentence. For this purpose, the paraphrased sentences resulting from the script "entailment_by_paraphrasing.py"
(to be found in "entailment_by_paraphrasing.csv") are used as new premise sentences and the original premise sentences are
used as new hypothesis sentences (if they contain additional information, e.g. new nouns or adjectives). In this way, additional
(neutral) information will be provided by the hypothesis sentences. The resulting dataframe contains the new premise sentences
in the first column and the neutral hypothesis sentences in the columns 2 to n.

What you need to do: install required packages and change the path according to your directory (line 71)
"""

import spacy
import pandas as pd
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)


# load spacy's language model "en_core_web_lg"
nlp_model = spacy.load('en_core_web_lg')


# Swaps the premise sentence with its paraphrased versions if it contains more information (e.g., new nouns/adjectives).
def extract_paraphrased_sents_for_neutralization (df, nlp_model):
    sentences = []
    sent_ct = 0
    columns = df.columns.tolist()

    for _, i in df.iterrows():
        col_ct = 0
        doc = nlp_model(str(i['sent']))
        adj_original = []
        nouns_prons_original = []
        for token in doc:
            if token.pos_ == 'NOUN' or token.pos_ == 'PRON':
               nouns_prons_original.append(str(token.lemma_).lower())
            if token.pos_ == 'ADJ' or token.pos_ == 'ADV':
                adj_original.append(str(token.lemma_).lower())

        if sent_ct % 100 == 0:
            print('processed sentences: ', sent_ct)

        for c in columns:
            if col_ct > 1:
                doc = nlp_model(str(i[c]))
                nouns_prons_new = []
                adj_new = []
                for token in doc:
                    if token.pos_ == 'NOUN' or token.pos_ == 'PRON':
                        nouns_prons_new.append(str(token.lemma_).lower())
                    if token.pos_ == 'ADJ' or token.pos_ == 'ADV':
                        adj_new.append(str(token.lemma_).lower())

                adj_new = list(set(adj_original)- set(adj_new))
                nouns_prons_new = list(set(nouns_prons_original) - set(nouns_prons_new))

                if len(str(i[c])) < len(str(i['sent'])):
                    div = len(str(i[c])) / len(str(i['sent']))
                    if div <= 0.75:
                        if adj_new and nouns_prons_new:
                            sent_couple = [str(i[c]), str(i['sent'])]
                            sentences.append(sent_couple)
            col_ct += 1
        sent_ct += 1
    return sentences



# load the paraphrased sentences
train_df = pd.read_csv('output/entailment_by_paraphrasing.csv', sep = ';')
train_df = train_df#[:2000]  # draw a sample, if desired
sents = extract_paraphrased_sents_for_neutralization (train_df, nlp_model)
neutral_df = pd.DataFrame(sents, columns = ['sent', 'neutral_by_paraphraze_switch'])
neutral_df = neutral_df.dropna()
neutral_by_paraphraze_switches = neutral_df.drop(neutral_df[neutral_df.sent == 'nan'].index)

# write final output to disc:
neutral_by_paraphraze_switches.to_csv('output/neutral_by_paraphrased_switch.csv')
