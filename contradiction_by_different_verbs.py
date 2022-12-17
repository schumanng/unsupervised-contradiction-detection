
"""
This script is used to create contradicting hypothesis sentences by changing verbs.
In this text generation approach, no direct antonyms of verbs are used, but verbs that differ sufficiently from
the original ones so that they represent a contradiction to the premise sentence in the context of the whole sentence.
For example, in the sentence "A boy is jumping on a skateboard", a contradictory hypothesis sentence can be created by
changing the verb: "A boy is lying on a skateboard". For this purpose, we have provided all verbs in the sentence with
"MASK tokens" and then passed the sentence to a Bert model, which tries to fill the masked words. The advantage of this
approach is that Bert predicts the masked words based on the surrounding words, which makes the inserted words fit well
within the sentence (also in terms of tense and numerus). Since the verbs should be as opposite as possible, a threshold
was set for the similarity to the original verb, which had to be undercut for the new verb to be inserted.
The modified sentences are then transformed into alternative sentences with the same meaning using a paraphrasing
model. The resulting dataframe contains the original premise sentences in the first column and the newly generated
contradicting hypothesis sentences in the columns 2 to n.

What you need to do: install required packages and change the path according to your directory (line 251 and 289)
"""

import spacy
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BertTokenizer, BertForMaskedLM
from sentence_transformers import SentenceTransformer, util

torch_device = 'cpu'
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', None)

# load Spacy-model for POS-Tagging und Dependency-Parsing
nlp_model = spacy.load('en_core_web_lg')

# load Pegasus-model for Paraphrasing
pegasus_model = AutoModelForSeq2SeqLM.from_pretrained("tuner007/pegasus_paraphrase")
pegasus_tokenizer = AutoTokenizer.from_pretrained("tuner007/pegasus_paraphrase")

# load BERT-model for Masked-Word-Predictions
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model_for_demasking = BertForMaskedLM.from_pretrained('bert-base-cased')
model_for_demasking.eval()

# load BERT-model for semantic sentence/word comparision
st_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')



# helper function to filter spans
def filter_spans(spans):
    get_sort_key = lambda span: (span.end - span.start, -span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        # Check for end - 1 here because boundaries are inclusive
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
        seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result



# searches for specified morph type in given list
def getMorphType(morph_list, morph_type):
    for item in morph_list:
        if morph_type in str(item):
            return item



# returns n paraphrased versions of a given sentence
def get_response(input_text, num_return_sequences, num_beams):
    batch = pegasus_tokenizer([input_text], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(
        torch_device)
    translated = pegasus_model.generate(**batch, max_length=60, num_beams=num_beams,
                                        num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = pegasus_tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text



# creates short sentences that focus on the new verbs
def extract_sub_sentences (nlp_model, sent):
    doc = nlp_model(sent)
    # Merge entities and noun chunks into one token
    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)

    new_sents = []
    sentence = []
    prep_subtrees = []

    # simplify the main sentence statement
    for token in doc:
        prep_subtree = []
        head_contains_prep = False
        w = token
        while w.dep_ != 'ROOT':
            w = w.head
            if w.dep_ == "prep":
                prep_subtree.append(str(w))
                head_contains_prep = True
        if token.dep_ == "prep":
            for sub_t in token.subtree:
                prep_subtree.append(str(sub_t.text))
            prep_subtree = ' '.join(prep_subtree)
            prep_subtree = prep_subtree.replace(' ,', ',')
            if prep_subtree[-2:] == ' .':
                prep_subtree = prep_subtree[:-2] + '.'
            prep_subtree = prep_subtree.replace('[ ', '')
            prep_subtree = prep_subtree.replace(' ]', '')
            prep_subtrees.append(prep_subtree)

        if token.dep_ != "prep" and not head_contains_prep:
            sentence.append(str(token))

    new_sent = ' '.join(sentence)
    new_sent = new_sent.replace(' ,', ',')
    if new_sent[-2:] == ' .':
        new_sent = new_sent[:-2] + '.'
    new_sent = new_sent.replace('[ ', '')
    new_sent = new_sent.replace(' ]', '')

    # check whether the resulting sentence is still a valid sentence
    doc = nlp_model(new_sent)
    valid = False
    if new_sent != sent:
        for token in doc:
            if token.pos_ == 'AUX' or token.pos_ == 'VERB':
                valid = True
        if valid:
            new_sents.append(new_sent)
    return new_sents



# exchanges verbs for "MASK-tokens"
def add_masked_labels_for_contradiction (doc):
    mask_added = False
    original_token = ''
    new_sent = []
    for token in doc:
        if not mask_added and (token.pos_ == 'VERB'):
            svp = [w for w in token.rights if w.dep_ == "svp"]
            if not svp:
                original_token = str(token)
                new_sent.append('[MASK]')
                mask_added = True
            else:
                new_sent.append(str(token))
        else:
            new_sent.append(str(token))
    new_sent = ' '.join(new_sent)
    new_sent = new_sent.replace(' ,', ',')
    new_sent = new_sent.replace(' :', ':')
    new_sent = new_sent.replace('( ', '(')
    new_sent = new_sent.replace(' )', ')')
    new_sent = new_sent.replace("'", '')
    new_sent = new_sent.lstrip().rstrip()
    if new_sent[-2:] == ' .':
        new_sent = new_sent[:-2] + '.'
    return new_sent, original_token



# predicts masked words (in this case verbs) within a sentence based on the surrounding words
def predict_masked_words_for_contradiction(text, nlp_model, top_k=20):
    # Tokenize input using spacy
    original_text = text
    doc = nlp_model(text)
    text, original_token = add_masked_labels_for_contradiction(doc)
    doc_as_list = list(nlp_model(text))
    text = "[CLS] %s [SEP]" % text
    tokenized_text = tokenizer.tokenize(text)
    masked_indices = [i for i, x in enumerate(tokenized_text) if x == "[MASK]"]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])

    new_modifiers = []
    for masked_index in masked_indices:
        # Predict all tokens
        with torch.no_grad():
            outputs = model_for_demasking(tokens_tensor)
            predictions = outputs[0]

        probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
        top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)

        for i, pred_idx in enumerate(top_k_indices):
            predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
            token_weight = top_k_weights[i]

            embeddings1 = st_model.encode([original_token], convert_to_tensor=True)
            embeddings2 = st_model.encode([predicted_token], convert_to_tensor=True)
            cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)
            similarity = round(cosine_score[0][0].item(), 2)

            # A new verb is only used if the similarity to the original verb is less than or equal to 0.45
            if (('#' not in predicted_token) and (len(predicted_token) > 5) and (similarity <= 0.45)
                    and (float(token_weight) >= 0.0005)):
                token_pos = nlp_model(str(predicted_token))
                if token_pos[0].pos_ == 'VERB':
                    new_modifiers.append(predicted_token)
                    break

    modifier_index = 0
    masked_indices = [i for i, x in enumerate(doc_as_list) if str(x) == "MASK"]

    new_sents = []
    if new_modifiers:
        for index in masked_indices:
            doc_as_list[index] = new_modifiers[modifier_index]
            modifier_index += 1

        doc_as_list = [str(i) for i in doc_as_list]
        new_sent = ' '.join(doc_as_list)
        new_sent = new_sent.replace(' ,', ',')
        new_sent = new_sent.replace(' ,', ',')
        new_sent = new_sent.replace(' :', ':')
        new_sent = new_sent.replace('( ', '(')
        new_sent = new_sent.replace(' )', ')')
        new_sent = new_sent.lstrip().rstrip()
        if new_sent[-2:] == ' .':
            new_sent = new_sent[:-2] + '.'
        new_sent = new_sent.replace('[ ', '')
        new_sent = new_sent.replace(' ]', '')
        new_sents.append(new_sent)

        sub_sent = extract_sub_sentences(nlp_model, new_sent)
        if sub_sent:
            if new_modifiers[0] in sub_sent[0]:
                new_sents.append(sub_sent[0])
        paraphrased = get_response(new_sent, 5, 5)
        for p_sent in paraphrased:
            if len(p_sent) < len(new_sent):
                div = len(p_sent) / len(new_sent)
                if (new_modifiers[0] in p_sent) and (p_sent != new_sent) and div <= 0.7:
                    new_sents.append(p_sent)

    elif not new_modifiers:
        new_sents.append(original_text)
    return new_sents




# load SNLI training set
train_df = pd.read_csv('input/snli_1.0_train.csv', sep=',')  # adjust the path according to your directory
train_df_sample = train_df#[:2000]  # draw a sample, if desired
train_df_sample = train_df_sample['sentence1']
train_df_sample = train_df_sample.drop_duplicates()
contradicting_verbs_inserted = []
changes_ct = 0
sent_ct = 0

for sent in train_df_sample:
    sent = sent.replace("'", "")
    sent_level_lst = [sent]
    changed = False
    new_sent = predict_masked_words_for_contradiction(sent, nlp_model)
    if new_sent:
        for n_sent in new_sent:
            if n_sent != sent:
                sent_level_lst.append(n_sent)
                changed = True
    if changed:
        changes_ct += 1
    contradicting_verbs_inserted.append(sent_level_lst)
    if sent_ct % 50 == 0:
        print('sentences processed: ', sent_ct + 1, ' | newly created sentences: ',
              changes_ct, ' | example of original sentence: ',
              sent, '| and its new neutral sentences: ', new_sent)
    sent_ct += 1
print('newly created sentences: ',changes_ct, ' Total sentences:', len(train_df_sample))


# function that turns the contradicting sentence-list into a dataframe
contradicting_verbs_inserted_df = pd.DataFrame(contradicting_verbs_inserted)
counter = 0
for col in contradicting_verbs_inserted_df.columns:
    contradicting_verbs_inserted_df.rename(columns={col:'different_verb_'+str(counter)},inplace=True)
    counter += 1
contradicting_verbs_df = contradicting_verbs_inserted_df.rename(columns={"different_verb_0": "sent"})

# write final output to disc:
contradicting_verbs_df.to_csv('output/contradiction_by_different_verbs.csv', sep=';')
