
"""
This script is used to create neutral hypothesis sentences by switching previously extracted sub-sentences with the
original premise sentence. For this purpose, the sub-sentences resulting from the script "entailment_by_subsentence.py"
(to be found in "entailment_by_subsentence.csv") are used as new premise sentences and the original premise sentences
are used as new hypothesis sentences. In this way, additional (neutral) information will be provided by the hypothesis 
sentences. The "new" hypothesis sentences are then also transformed into alternative sentences with the same meaning using 
a paraphrasing model. The resulting dataframe contains the premise sentences in the first column and the newly generated
neutral hypothesis sentences in the columns 2 to n.

What you need to do: install required packages and change the path according to your directory (line 41)
"""

import spacy
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

torch_device = 'cpu'
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', None)

# load language models
nlp_model = spacy.load('en_core_web_lg')
pegasus_model = AutoModelForSeq2SeqLM.from_pretrained("tuner007/pegasus_paraphrase")
pegasus_tokenizer = AutoTokenizer.from_pretrained("tuner007/pegasus_paraphrase")



# returns n paraphrased versions of a given sentence
def get_response(input_text, num_return_sequences, num_beams):
    batch = pegasus_tokenizer([input_text], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(
        torch_device)
    translated = pegasus_model.generate(**batch, max_length=60, num_beams=num_beams,
                                        num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = pegasus_tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text



# load the entailment_by_subsentence.csv
entailment_by_subsentence = pd.read_csv('output/entailment_by_subsentence.csv', sep = ';')
entailment_by_subsentence = entailment_by_subsentence.dropna()
entailment_by_subsentence = entailment_by_subsentence[['sub_sent_1', 'sent']]


switched_sub_sentences_lst = []
for index, row in entailment_by_subsentence.iterrows():
    sent_level_lst = [row['sub_sent_1'], row['sent']]
    doc = nlp_model(row['sent'])
    nouns_prons_original = []
    adj_original = []
    verb_original = ''
    for token in doc:
        if token.pos_ == 'NOUN' or token.pos_ == 'PRON':
            nouns_prons_original.append(str(token).lower())
        if token.pos_ == 'ADJ' or token.pos_ == 'ADV':
            adj_original.append(str(token).lower())
        if token.pos_ == 'VERB':
            verb_original = token.lemma_
    paraphrased = get_response(row['sent'], 10, 10)
    counter = 1
    for p_sent in paraphrased:
        if counter == 6:
            break
        doc = nlp_model(p_sent)
        nouns_prons_new = []
        adj_new = []
        verb_new = ''
        for token in doc:
            if token.pos_ == 'NOUN' or token.pos_ == 'PRON':
                nouns_prons_new.append(str(token).lower())
            if token.pos_ == 'ADJ' or token.pos_ == 'ADV':
                adj_new.append(str(token).lower())
            if token.pos_ == 'VERB':
                verb_new = token.lemma_
        adj_new = list(set(adj_new) - set(adj_original))
        nouns_prons_new = list(set(nouns_prons_new) - set(nouns_prons_original))
        if verb_new == verb_original:
            if adj_new or nouns_prons_new:
                if (any(n_adj in p_sent for n_adj in adj_new)) or (any(n_noun in p_sent for n_noun in nouns_prons_new)):
                    if len(p_sent) > len(row['sub_sent_1']):
                        div = len(row['sub_sent_1']) / len(p_sent)
                        if (div <= 0.75) and p_sent != row['sent']:
                            sent_level_lst.append(p_sent)
                            counter += 1

    if index % 25 == 0:
        print('processed sentences: ',
              index, ' | premise sentence (1st item) and the newly added neutral hypothesis sencences (items 2 to n): ',
              sent_level_lst)
    switched_sub_sentences_lst.append(sent_level_lst)




# turns the neutral sentence-list into a pandas dataframe
switched_sub_sentences_lst_df = pd.DataFrame(switched_sub_sentences_lst)
counter = 0
for col in switched_sub_sentences_lst_df.columns:
    switched_sub_sentences_lst_df.rename(columns={col:'neutral_by_switching_'+str(counter)},inplace=True)
    counter += 1
switched_sub_sentences_lst_df = switched_sub_sentences_lst_df.rename(columns={"neutral_by_switching_0": "sent"})

# write final output to disc:
switched_sub_sentences_lst_df.to_csv('neutral_by_subsentence_switch.csv', sep=';')
