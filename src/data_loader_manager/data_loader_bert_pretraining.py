import json
import os
import random

from data_loader_manager.data_loader_wrapper import DataLoaderWrapper
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling


class DataLoaderBertPretrain(DataLoaderWrapper):

    def __init__(self, config, tokenizer):
        DataLoaderWrapper.__init__(self, config)

        self.tokenizer = tokenizer

    def LoadPretrainingTweets(self, module_config):

        train_sentences = []
        eval_sentences = []

        i = 0
        for year in range(2019, 2022+1):
            print(f"Searching in TongueSwitcher for year {year}")
            for month in range(1, 12+1):
                print(f"Searching in TongueSwitcher for month {month}")
                with tqdm(total=12) as pbar:

                    file_path = f'{self.config.tongueswitcher_corpus}/annotation_{year}_{str(month).zfill(2)}.jsonl'

                    if not os.path.isfile(file_path):
                        continue

                    with open(file_path, 'r') as file:
                        for line in tqdm(file):
                            json_line = json.loads(line)
                            train_sentences.append(json_line["text"])
                            i += 1

        i = 0
        for year in range(2023, 2023+1):
            print(f"Searching in TongueSwitcher for year {year}")
            for month in range(1, 2+1):
                print(f"Searching in TongueSwitcher for month {month}")
                with tqdm(total=12) as pbar:

                    file_path = f'{self.config.tongueswitcher_corpus}/annotation_{year}_{str(month).zfill(2)}.jsonl'

                    if not os.path.isfile(file_path):
                        continue

                    with open(file_path, 'r') as file:
                        for line in tqdm(file):
                            json_line = json.loads(line)
                            eval_sentences.append(json_line["text"])
                            i += 1

        print(f"Found {len(train_sentences)} code-switching tweets to train mBERT model on")
        print(f"and {len(eval_sentences)} code-switching tweets to eval mBERT model on")

        # Shuffle the sentences randomly
        random.shuffle(train_sentences)

        # Tokenize train and eval datasets
        train_tokenized_datasets = [self.tokenizer(sentence) for sentence in train_sentences]
        eval_tokenized_datasets = [self.tokenizer(sentence) for sentence in eval_sentences]

        # Data collator for MLM
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)

        self.data.data_collator = data_collator
        self.data.train_tokenized_datasets = train_tokenized_datasets
        self.data.eval_tokenized_datasets = eval_tokenized_datasets