# Unsupervised-NLI

(Partial) implementation of different text generation approaches to create a training base that can be used in an unsupervised NLI setting. The basic idea and functionality is taken from Varshney et al. (2022) "[Unsupervised Natural Language Inference Using PHL Triplet Generation, ACL 2022](https://arxiv.org/abs/2110.08438)", but some of the individual text generation features were modified and extended by me (e.g. by using masked-word-prediction with the help of BERT, sentence- and word similarity or also GPT2-text-generation). Before I made my own modifications, I rebuilt the text features based on the descriptions from the paper, because at this point in time (17 December 2022) Varshney's code has not yet been published and I wanted to test the approach for a few experiments on domain data. However, within this repository, the SNLI training dataset is used for text generation.

To generate the sentence pairs, the 14 scripts can be started individually and independently of each other. Only in the case of the scripts "neutral_by_paraphrased_switch.py" and "neutral_by_subsentence_switch.py" the scripts "entailment_by_paraphrasing.py" and "entailment_by_subsentence.py" must be executed beforehand.

Link to Varshney's Repo: https://github.com/nrjvarshney/unsupervised_NLI

## Required packages:
- re
- wordnet
- transformers
- sentence-transformers
- numpy
- pandas
- random
- spacy
- torch
