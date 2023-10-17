
import json

from tongueswitcher.tongueswitcher_executor import TSExecutor
from tongueswitcher.word_level_processor import Processor


class TongueSwitcher(TSExecutor, Processor):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)    

        self.bigrams = self.open_json(config.bigram_data_dir)
        self.clh_pos_tags = self.open_json(config.crosslingual_homographs_dir)

    def tongueswitcher_detect(self, annotated_words):

        annotated_words = self.word_processor(annotated_words)
        annotated_words = self.first_smoothen(annotated_words)      
        annotated_words = self.ngram_search(annotated_words, self.bigrams)
        annotated_words = self.second_smoothen(annotated_words)   

        annotated_words = self.add_bio_keys(annotated_words)   

        return annotated_words

    @staticmethod
    def add_bio_keys(tweet_dict):
        last_lan = None
        for token in tweet_dict:
            if token["lan"] != 'E':
                token["bio"] = 'O'
            elif token["lan"] == last_lan:
                token["bio"] = 'I-' + token["lan"]
            else:
                token["bio"] = 'B-' + token["lan"]
            last_lan = token["lan"]

        return tweet_dict

    @staticmethod
    def flatten_morph(l):
        return ['+'.join(item) for item in l]

    @staticmethod
    def flatten_labels(l):
        return [''.join(item) for item in l]

    @staticmethod
    def flatten(input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def second_smoothen(self, annotated_words): 


        for i in range(len(annotated_words)):

            if annotated_words[i]["lan"] != 'U':
                continue

            if i == 0:
                if len(annotated_words) > 1 and annotated_words[1]["lan"] != 'U' and annotated_words[1]["lan"] != 'M':
                   annotated_words[i]["lan"] = annotated_words[1]["lan"]
                else:
                    annotated_words[i]["lan"] = "D"
            elif i + 1 < len(annotated_words) - 1 and (annotated_words[i+1]["lan"] == 'D' or annotated_words[i+1]["lan"] == 'E'):
                annotated_words[i]["lan"] = annotated_words[i+1]["lan"]
            elif annotated_words[i-1]["lan"] == 'D' or annotated_words[i-1]["lan"] == 'E':
                annotated_words[i]["lan"] = annotated_words[i-1]["lan"]
            elif annotated_words[i-1]["lan"] == 'M':
                annotated_words[i]["lan"] = annotated_words[i-1]["lans"][-1]
            elif i + 1 < len(annotated_words) - 1 and (annotated_words[i+1]["lan"] == 'M'):
                annotated_words[i]["lan"] = annotated_words[i+1]["lans"][0]
            else:
                annotated_words[i]["lan"] = "D"
                    
        return annotated_words
    
    def first_smoothen(self, annotated_words): 


        for i in range(len(annotated_words)):

            if annotated_words[i]["lan"] != 'U':
                continue
            elif i == 0 or i == len(annotated_words) - 1:
                continue
            if annotated_words[i-1]["lan"] == annotated_words[i+1]["lan"]:
                annotated_words[i]["lan"] = annotated_words[i-1]["lan"]
        
        return annotated_words

    def ngram_search(self, annotated_words, ngrams):

        words = [i["token"].lower() for i in annotated_words]

        for i in range(len(annotated_words)):
            if annotated_words[i]["lan"] == 'U' and annotated_words[i]["token"].lower() in ngrams:
                for ng, _, lan in ngrams[annotated_words[i]["token"].lower()]:
                    phrase = ng.split()
                    pivot = phrase.index(annotated_words[i]["token"].lower())
                    exp_len = len(phrase)

                    if i-pivot+exp_len <= len(words) and i-pivot >= 0 and ' '.join(words[i-pivot:i-pivot+exp_len]) == ng.lower():
                        annotated_words[i]["lan"] = lan
                        annotated_words[i]["step"] = 11
                        break

        return annotated_words

    @staticmethod
    def open_json(dir):
        with open(dir, 'r', encoding = 'utf-8') as f:
            return json.load(f)
