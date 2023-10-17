import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm

DICTIONARIES_PATH = '../data/cache/dictionaries.pkl'
CACHE_FILE = Path('')

def tag_corpus_with_pos(corpus_sentences, tagger, batch_size):
    print("Tagging POS for the corpus...")
    tagged_sentences = []
    for i in tqdm(range(0, len(corpus_sentences), batch_size)):
        batch = corpus_sentences[i:i+batch_size]
        tagger.predict(batch, mini_batch_size=64)
        tagged_sentences.extend(batch)
    return tagged_sentences

def process_corpus(input_file, lang_code):
    print(f"Processing {lang_code} corpus and preparing sentences...")
    sentences = []
    with open(input_file, "r", encoding="utf-8") as f:
        corpus = csv.reader(f, delimiter="\t")
        for i, row in enumerate(corpus):
            sentences.append(row[1])
    
    print(f"Total sentences in {lang_code} corpus: {len(sentences)}")
    
    return [Sentence(sentence_text) for sentence_text in tqdm(sentences, desc=f"Preparing {lang_code} sentences")]

def save_to_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data saved to {output_file}")

en_input_file = "../data/wordlist_data/eng_news_2020_1M-sentences.txt"
de_input_file = "../data/wordlist_data/deu_news_2021_1M-sentences.txt"

output_file = "../data/crosslingual_homographs/clh_pos_tags.json"

if CACHE_FILE.exists():
    with open(CACHE_FILE, "rb") as f:
        combined_pos_data = pickle.load(f)
else:
    print("Loading SequenceTagger model...")
    tagger = SequenceTagger.load("flair/upos-multi")

    en_corpus_sentences = process_corpus(en_input_file, 'E')
    de_corpus_sentences = process_corpus(de_input_file, 'D')

    batch_size = 10000

    # Tagging POS for English corpus
    en_tagged_corpus = tag_corpus_with_pos(en_corpus_sentences, tagger, batch_size)
    print("Tagging POS for German corpus...")
    de_tagged_corpus = tag_corpus_with_pos(de_corpus_sentences, tagger, batch_size)

    combined_pos_data = defaultdict(list)
    for en_sentence, de_sentence in tqdm(zip(en_tagged_corpus, de_tagged_corpus), desc="Combining POS data"):
        for en_token, de_token in zip(en_sentence, de_sentence):
            en_word = en_token.text
            en_pos = en_token.get_label("upos").value
            de_word = de_token.text
            de_pos = de_token.get_label("upos").value

            if en_word.isalpha() and len(en_word) > 2: combined_pos_data[en_word.lower()].append((en_pos, 'E'))
            if de_word.isalpha() and len(de_word) > 2: combined_pos_data[de_word.lower()].append((de_pos, 'D'))

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(combined_pos_data, f)

with open(DICTIONARIES_PATH, 'rb') as f:
    dictionaries = pickle.load(f)

final_pos_data = defaultdict(list)

combined_pos_data = {"handy": combined_pos_data["handy"]}

for word, pos_list in tqdm(combined_pos_data.items(), desc="Processing combined data"):

    pos_counter = Counter((pos, pos_lang) for pos, pos_lang in pos_list)
    

    if word == "handy":
        print("WOW")


    pos_freq_map = {(pos, pos_lang): freq for (pos, pos_lang), freq in pos_counter.items() if (freq > 2 and pos != 'X')}.copy()
    other_lan = {"E": "D", "D": "E"}
    skip = False

    for (pos, pos_lang), freq in pos_freq_map.copy().items():
        if freq < 10 and (pos, pos_lang) in pos_freq_map:
            del pos_freq_map[(pos, pos_lang)]

    for (pos, pos_lang), freq in pos_freq_map.copy().items():
        if (pos, other_lan[pos_lang]) in pos_freq_map:
            other_freq = pos_freq_map[(pos, other_lan[pos_lang])]
            if (freq > 100 and other_freq > 100) or abs(freq - other_freq) < 10:
            #     print(f"Identical: {word}")
            #     skip = True
            #     break
            # elif abs(freq - other_freq) < 10:
                del pos_freq_map[(pos, other_lan[pos_lang])]
                if (pos, pos_lang) in pos_freq_map:
                    del pos_freq_map[(pos, pos_lang)]              
            elif freq >= other_freq:
                del pos_freq_map[(pos, other_lan[pos_lang])]
                # print(f"Mistake: {word}")

    
    pos_s = [i[0] for i in pos_freq_map.keys()]
    if len(pos_freq_map) == 2 and 'NOUN' in pos_s and 'PROPN' in pos_s:
        continue

    if 'NUM' in pos_s:
        continue

    other_pos = {"NOUN": "PROPN", "PROPN": "NOUN"}

    for (pos_1, lan_1), freq_1 in pos_freq_map.copy().items():
        if pos_1 == 'NOUN' or pos_1 == 'PROPN':
            for (pos_2, lan_2), freq_2 in pos_freq_map.copy().items():
                if pos_2 == other_pos[pos_1] and lan_2 == other_lan[lan_1]:
                    if freq_1 > 10*freq_2:
                        if (pos_2, lan_2) in pos_freq_map:
                            del pos_freq_map[(pos_2, lan_2)]
                    elif freq_2 > 10*freq_1:
                        if (pos_1, lan_1) in pos_freq_map:
                            del pos_freq_map[(pos_1, lan_1)]
                    else:
                        if (pos_2, lan_2) in pos_freq_map:
                            del pos_freq_map[(pos_2, lan_2)]
                        if (pos_1, lan_1) in pos_freq_map:
                            del pos_freq_map[(pos_1, lan_1)]

    if skip:
        continue

    lans = [i[1] for i in pos_freq_map.keys()]
    if len(set(lans)) < 2:
        continue

    if word in dictionaries['cs']:
        print(f"Removing as in dictionaries: {word}")
        continue
    
    final_pos_data[word] = {pos: pos_lang for (pos, pos_lang), freq in pos_freq_map.items()}

print(f"Number of CLH: {len(final_pos_data)}")
print(final_pos_data.keys())

# save_to_json(final_pos_data, output_file)

# print(f"Done! Combined POS data saved to {output_file}.")
