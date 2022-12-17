
"""
This script is used to create entailing hypothesis sentences using paraphrasing.
The resulting dataframe contains the original premise sentences in the first column and the newly generated
entailing hypothesis sentences in the columns 2 to n.

What you need to do: install required packages and change the path according to your directory (line 35)
"""

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

torch_device = 'cpu'
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', None)

# load pegasus model for paraphrasing
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



# load SNLI training set
train_df = pd.read_csv('input/snli_1.0_train.csv', sep=',')  # adjust the path according to your directory
train_df_sample = train_df[:20]  # draw a sample, if desired
train_df_sample = train_df_sample['sentence1']
train_df_sample = train_df_sample.drop_duplicates()
paraphrased_sentences_lst = []
changes_ct = 0
sent_ct = 0

print('start paraphrasing')
for sent in train_df_sample:
    sent_level_lst = [sent]
    new_sent = get_response(sent, 10, 10)
    changed = False
    if new_sent:
        for n_sent in new_sent:
            if n_sent != sent:
                sent_level_lst.append(n_sent)
                changed = True
    if changed:
        changes_ct += 1
    paraphrased_sentences_lst.append(sent_level_lst)
    if sent_ct % 50 == 0:
        print('sentences processed: ', sent_ct + 1, ' | newly created sentences: ',
              changes_ct, ' | example of original sentence: ',
              sent, '| and its new paraphrased sentences: ', new_sent)
    sent_ct += 1


# turns the entailing sentence-list into a pandas dataframe
paraphrases_df = pd.DataFrame(paraphrased_sentences_lst)
counter = 0
for col in paraphrases_df.columns:
    paraphrases_df.rename(columns={col:'paraphrased_'+str(counter)},inplace=True)
    counter += 1
paraphrases_df = paraphrases_df.rename(columns={"paraphrased_0": "sent"})

# write final output to disc:
paraphrases_df.to_csv('output/entailment_by_paraphrasing.csv', sep=';')
