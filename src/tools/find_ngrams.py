import csv
import json
from collections import Counter

from nltk import ngrams
from tqdm import tqdm


def find_common_ngrams(language_code, input_file, n):
    # Step 1: Tokenize the sentences
    print(f"Loading {language_code} corpus and tokenizing sentences...")
    sentences = []
    with open(input_file, "r", encoding="utf-8") as f:
        corpus = csv.reader(f, delimiter="\t")

        for i, row in enumerate(corpus):
            sentences.append(row[1].lower().split())

    # Step 2: Find the most common n-grams
    print(f"Finding the most common {n}-grams for {language_code}...")
    all_ngrams = []
    for tokens in tqdm(sentences):
        grams = list(ngrams(tokens, n))
        all_ngrams.extend(grams)

    # Count the occurrences of each n-gram
    ngram_counter = Counter(all_ngrams)

    # Create a dictionary to store the bigrams for each word
    word_ngrams_dict = {}
    for ngram, count in tqdm(ngram_counter.most_common(10000)):
        words = list(ngram)
        ngram_key = " ".join(words)
        bigram_data = (ngram_key, count, language_code)

        for word in words:
            if word not in word_ngrams_dict:
                word_ngrams_dict[word] = []
            word_ngrams_dict[word].append(bigram_data)

    return word_ngrams_dict

def merge_dictionaries(dict_list):
    combined_dict = {}
    for d in dict_list:
        for key, value in d.items():
            if key in combined_dict:
                combined_dict[key].extend(value)
            else:
                combined_dict[key] = value
    return combined_dict

def save_to_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# English
en_input_file = "../data/wordlist_data/eng_news_2020_1M-sentences.txt"
de_input_file = "../data/wordlist_data/deu_news_2021_1M-sentences.txt"

output_file = "../data/bigrams/combined_bigrams.json"

en_bigram_data = find_common_ngrams('E', en_input_file, n=2)
de_bigram_data = find_common_ngrams('D', de_input_file, n=2)

en_bigrams = set()
de_bigrams = set()

combined_word_bigram_dict = merge_dictionaries([en_bigram_data, de_bigram_data])
for word, bigram_data in combined_word_bigram_dict.items():
    combined_word_bigram_dict[word] = sorted(bigram_data, key=lambda x: x[1], reverse=True)

    for bigram, freq, lan in bigram_data:
        if lan == 'E':
            en_bigrams.add(bigram)
        elif lan == 'D':
            de_bigrams.add(bigram)

both_bigrams = en_bigrams.intersection(de_bigrams)

print(both_bigrams)
print(f"Number of matching bigrams: {len(both_bigrams)}")

# Save the combined results to a JSON file
save_to_json(combined_word_bigram_dict, output_file)

print(f"Done! Combined bigram data saved to {output_file}.")
