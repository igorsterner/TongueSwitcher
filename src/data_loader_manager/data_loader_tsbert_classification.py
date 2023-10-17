import re

import torch
from data_loader_manager.data_loader_wrapper import DataLoaderWrapper
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import DataCollatorForTokenClassification
from utils.corpus import Corpus

from tokenizations import get_alignments


class ClassificationDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

class DataLoaderBertClassification(DataLoaderWrapper):

    def __init__(self, config, tokenizer):
        DataLoaderWrapper.__init__(self, config)

        self.tokenizer = tokenizer

    def LoadClassificationTweets(self, module_config):

        train_tokens, train_labels = [], []
        train_subword_labels = []

        corpus = Corpus(self.config.denglisch_corpus)
        toks, tags = corpus.get_sentences()
        for sent_toks, sent_tags in tqdm(zip(toks, tags)):

            if len(sent_toks) > 100:
                continue

            sent_toks = [t.replace("’", "'").replace("”", "'").replace("“", "'").replace("„", "'").replace("―", "-").replace("–", "-").replace("…", "...").replace("`", "'").replace("‘", "'").replace("—", "-").replace("´", "'").replace("'¯\\ \\_(ツ)_/¯'", '!').replace("¯", "-") for t in sent_toks]
            sent_toks = self.replace_emojis_with_X(sent_toks)
            
            subword_ids = self.tokenizer(sent_toks, is_split_into_words=True)
            subword_tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in subword_ids["input_ids"]]
            subword_labels = self.get_subword_labels(sent_toks, subword_tokens, sent_tags)

            assert len(subword_labels) == len(subword_tokens)

            train_subword_labels.append(subword_labels)
            train_labels.append(sent_tags)
            train_tokens.append(sent_toks)
        
        train_encodings = self.tokenizer(train_tokens, is_split_into_words=True)
        train_encodings = {'input_ids': train_encodings['input_ids'], 'labels': train_subword_labels}

        label2id = {'D': 0, 'M': 1, 'E': 2, 'O': 3, 'SE': 4, 'SD': 5, 'SO': 6}

        train_encodings['labels'] = [[label2id[t] for t in sent] for sent in train_encodings['labels']]

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        self.data.data_collator = data_collator
        self.data.train_encodings = ClassificationDataset(train_encodings)

    @staticmethod
    def get_subword_labels(a, b, a_labels):
        a2b, b2a = get_alignments(a, b)

        # Assign labels to subwords
        b_labels = []
        most_common = 'D'

        for i, label_indices in enumerate(b2a):

            aligned_subwords = []

            if label_indices:
                for j in label_indices:
                    if j < len(a_labels):
                        aligned_subwords.append(a_labels[j])

            if not aligned_subwords:
                aligned_subwords = [most_common]

            most_common = max(set(aligned_subwords), key=aligned_subwords.count)

            b_labels.append(most_common)
        
        return b_labels
    
    @staticmethod
    def replace_emojis_with_X(tokens):
        emoj = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002500-\U00002BEF"  # chinese char
            u"\U00002702-\U000027B0"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
                        "]+", re.UNICODE)
        return ['X' if re.match(emoj, token) else token for token in tokens]