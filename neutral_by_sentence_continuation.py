
"""
This script is used to create neutral hypothesis sentences by continuing the premise sentence with further information.
For this purpose, the full stop at the end of each sentence is removed and then the sentence is passed to a gpt2 model
to "complete" the sentence. In a second step, it is then checked which of the new information generated in this way can
be used for a neutral hypothesis sentence (e.g. if there is at least one new noun or adjective in the sentence).
If corresponding sentences have been found, they are converted into alternative sentences with the same meaning using
a paraphrasing model. The resulting dataframe contains the original premise sentences in the first column and the newly
generated contradictory hypothesis sentences in the columns 2 to n.

What you need to do: install required packages and change the path according to your directory (line 23 and 152)
"""

import spacy
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, set_seed

torch_device = 'cpu'
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', None)

# load SNLI training set
train_df = pd.read_csv('input/snli_1.0_train.csv', sep=',')  # adjust the path according to your directory

# load spacy's language model "en_core_web_lg" and pegasus model for paraphrasing
pegasus_model = AutoModelForSeq2SeqLM.from_pretrained("tuner007/pegasus_paraphrase")
pegasus_tokenizer = AutoTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
nlp_model = spacy.load('en_core_web_lg')


# initialize gpt2 text-generation
generator = pipeline('text-generation', model='gpt2')
set_seed(42)


# clean the training data
def data_cleansing (df):
    df = df.dropna()
    df = df[(df['sentence1'].str.split().str.len() > 0) & (df['sentence2'].str.split().str.len() > 0)]
    df = df.drop(df[df['gold_label'].map(len) < 7].index)
    return df


# select premise sentence and remove duplicates
train_df = data_cleansing(train_df)
train_df = train_df['sentence1']
train_df = train_df.drop_duplicates()


# Remove (if present) the full stop at the end of each sentence and then pass the sentence to gpt2.
# In this way, the sentence is considered incomplete and is continued by the gpt2 accordingly.
# This loop can take up to 3 days (without a gpu).
print('start text generation using gpt2')
idx = 0
sents = []
for sent in train_df:
    if sent[-1:] == '.':
        sent = sent[:-1]
    quad = [sent]
    sent = generator(sent, max_length=35, num_return_sequences=3)
    item_0 = sent[0]
    item_1 = sent[1]
    item_2 = sent[2]
    quad.append(item_0['generated_text'])
    quad.append(item_1['generated_text'])
    quad.append(item_2['generated_text'])
    sents.append(quad)
    idx += 1
    if idx % 500 == 0:
        print('processed sentences: ', idx)
    if idx == 10:
        break


# create pandas dataframe and write it to disc
sents_df = pd.DataFrame(sents)
sents_df.to_csv('output/gpt-2-output.csv', index=False, sep=';')
print('sentence completion through GPT2 finished!')
#==================================================



# returns n paraphrased versions of a given sentence
def get_response(input_text, num_return_sequences, num_beams):
    batch = pegasus_tokenizer([input_text], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(
        torch_device)
    translated = pegasus_model.generate(**batch, max_length=60, num_beams=num_beams,
                                        num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = pegasus_tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text



# search for sentences where the new information added by gpt2 is "neutral"
def extract_valid_sentence_from_gpt2_output (df, nlp_model):
    sentences = []
    sent_ct = 0
    for index, row in df.iterrows():
        sents = [row['0'] + '.']
        col = 1
        while col <= 3:
            valid_sent = ''
            valid_sent_bool = False
            for char in row[str(col)]:
                valid_sent = valid_sent + char
                if valid_sent[-2:] == '. ':
                    valid_sent_bool = True
                    break
            if valid_sent[-2:] == '. ':
                valid_sent = valid_sent[:-2] + '.'
            if valid_sent_bool and valid_sent != sents[0]:
                doc_og = nlp_model(sents[0])
                doc_new = nlp_model(valid_sent)
                if len(doc_new) > len(doc_og):
                    diff = len(doc_new) - len(doc_og)
                    if (diff > 1) and (diff <= 6):
                        idx = 0
                        last_word = str(doc_og[len(doc_og)-2])
                        old_noun_or_adj = []
                        new_noun_or_adj = []
                        for token in doc_og:
                            if (token.pos_ == 'NOUN' and token.dep_ != "sb") or token.pos_ == 'ADJ':
                                old_noun_or_adj.append(str(token))
                        for token in doc_new:
                            if idx >= len(doc_og)-1:
                                if (token.pos_ == 'NOUN' and token.dep_ != "sb") or token.pos_ == 'ADJ':
                                    new_noun_or_adj.append(str(token))
                            idx += 1
                        if new_noun_or_adj:
                            new_noun_or_adj = list(set(new_noun_or_adj) - set(old_noun_or_adj))
                            if new_noun_or_adj:
                                sents.append(valid_sent)
                                paraphrased = get_response(valid_sent, 5, 5)
                                for p_sent in paraphrased:
                                    if (last_word in p_sent) and (len(p_sent) < len(valid_sent)):
                                        includes_new_nouns_or_adj = False
                                        for noun_or_adj in new_noun_or_adj:
                                            if noun_or_adj in p_sent:
                                                includes_new_nouns_or_adj = True
                                        if includes_new_nouns_or_adj:
                                            sents.append(p_sent)
            col += 1
        sentences.append(sents)
        if sent_ct % 100 == 0:
            print('processed sentences: ', sent_ct)
        sent_ct += 1

    return sentences


# load the already-generated sentences from csv-file and run the final extraction function
gpt_2_outout = pd.read_csv('output/gpt-2-output.csv', sep = ';')
neutral_noised_sentences_lst = extract_valid_sentence_from_gpt2_output(gpt_2_outout, nlp_model)

# turns the neutral sentence-list into a pandas dataframe
print('start neutral_sentences_by_GPT2')
neutral_noised_sentences_df = pd.DataFrame(neutral_noised_sentences_lst)
counter = 0
for col in neutral_noised_sentences_df.columns:
    neutral_noised_sentences_df.rename(columns={col:'neutral_by_GPT_'+str(counter)},inplace=True)
    counter += 1
neutral_noised_sentences_df = neutral_noised_sentences_df.rename(columns={"neutral_by_GPT_0": "sent"})

# write final output to disc:
neutral_noised_sentences_df.to_csv('output/neutral_by_sentence_continuation.csv', sep=';')
