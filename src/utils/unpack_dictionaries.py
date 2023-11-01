import csv
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

import requests
from tqdm import tqdm


def leipzig(dict_path, word_length=2, min_freq=5):

    words = []

    with open(dict_path, 'r', newline='', encoding='utf-8') as f:
        doc = csv.reader(f, delimiter='\t')

        for line in doc:
            phrase = line[1].lower()
            phrase = phrase.replace('-', ' ')
            for word in phrase.split():
                if (int(line[2]) >= min_freq) and word.isalpha() and (len(word) > word_length):
                    words.append(word)

    return set(words)

def dictcc(dict_path, min_freq=0):

    words = []

    with open(dict_path, 'r', newline='', encoding='utf-8') as f:
        doc = csv.reader(f, delimiter='\t')
        for line in doc:

            if len(line) == 0 or line[0][0] == '#':
                continue

            phrase = re.sub("[\(\[\{\<].*?[\)\]\}\>]", "", line[0])
            phrase = ''.join([i for i in phrase if (i.isalpha() or i == ' ' or i == "-" or i == ".")])
            phrase = phrase.lower().strip()

            if not phrase:
                continue
            elif phrase[-1] == '-':
                continue
            elif len(phrase.split()) > 1:
                continue
            else:
                words.append(phrase)


            # if len(line) == 0 or line[0][0] == '#':
            #     continue

            # phrase = line[0]

            # phrase = ''.join([i for i in phrase if (i.isalpha() or i == ' ')])

            # for word in phrase.split():
            #     words.append(word.lower())
    
    return words

def dictcc_multiword(dict_path, lan):

    multiwords = []

    with open(dict_path, 'r', newline='', encoding='utf-8') as f:
        doc = csv.reader(f, delimiter='\t')
        for line in tqdm(doc):

            if len(line) == 0 or line[0][0] == '#':
                continue
            
            if lan == 'de':
                phrase = re.sub("[\(\[\{\<].*?[\)\]\}\>]", "", line[0])
            elif lan == 'en':
                phrase = re.sub("[\(\[\{\<].*?[\)\]\}\>]", "", line[1])

            if '/' in phrase:
                phrase = phrase.split('/')[0].strip()
            phrase = ''.join([i for i in phrase if (i.isalpha() or i == ' ')])
            phrase = phrase.lower().strip()

            if not phrase:
                continue
            elif len(phrase.split()) < 2:
                continue
            else:
                multiwords.append(' '.join(phrase.split()))
    
    return multiwords

def dictcc_homographs(dict_path, gpt_borrow):

    german_dict = defaultdict(list)
    english_dict = defaultdict(list)

    with open(dict_path, 'r', newline='', encoding='utf-8') as f:
        doc = csv.reader(f, delimiter='\t')
        
        for line in tqdm(doc):
            if len(line) == 0 or line[0][0] == '#':
                continue
            
            if '<' in line[0]:
                if '.' not in line[0][line[0].find("<")+1:line[0].find(">")]:
                    continue
            
            if '<' in line[1]:
                if '.' not in line[1][line[1].find("<")+1:line[1].find(">")]:
                    continue

            german = re.sub("[\(\[\{\<].*?[\)\]\}\>]", "", line[0])
            german = ''.join([i for i in german if (i.isalpha() or i == ' ')])
            german = german.lower().strip()



            english = re.sub("[\(\[\\{\<].*?[\)\]\}\>]", "", line[1])
            english = ''.join([i for i in english if (i.isalpha() or i == ' ')])
            english = english.lower().strip()

            if (not english) or (not german):
                continue

            german_dict[german].append(english)
            english_dict[english].append(german)

            english_dict[english.replace(' ', '')].append(german.replace(' ', ''))
                
            if english[-1] == 's':
                english_dict[english[:-1]].append(german)

            if len(german.split()) > 1:
                for w in german.split():
                    english_dict[english].append(w)

            if len(english.split()) > 1:
                for w in english.split():
                    german_dict[german].append(w)

            
    
    homographs = {germ for germ in german_dict.keys() if (len(germ) > 2) and (germ in english_dict) and (germ not in english_dict[germ])}
    homographs = {eng for eng in homographs if (len(eng) > 2) and (eng in german_dict) and (eng not in german_dict[eng])}

    homographs_gpt = [w for w in homographs if (w in gpt_borrow) and ((gpt_borrow[w] == 'german') or (gpt_borrow[w] == 'english'))]

    gpt_borrow_ff = {}

    for word, lan in gpt_borrow.items():
        if word in homographs_gpt:
            gpt_borrow_ff[word] = 'mixed'
        else:
            gpt_borrow_ff[word] = lan
    
    return gpt_borrow_ff

def urban(dict_path):

    words = []

    entries = ""

    for letter in string.ascii_uppercase:
        with open(Path(dict_path) / f"{letter}.data", 'r', encoding='utf-8') as f:
            entries += f.read().replace('\n', ' ').replace('"', '')

    words = entries.split()

    counter = Counter(words)

    top_words = counter.items()
    urban = {}
    for word in top_words:
        low_word = word[0].lower()
        repeated = False
        # Remove words with three or more consecutively repeating letters
        # for i in range(len(low_word) - 2):
        #     if low_word[i] == low_word[i + 1] == low_word[i + 2]:
        #         repeated = True

        if (not repeated) and (word[1] > 10) and (word[0].isalpha()) and (len(word[0]) > 2) and (
                not word[0][0].isupper()):
            if low_word in urban:
                urban[low_word] += word[1]
            else:
                urban[low_word] = word[1]
    urban = {k: v for k, v in sorted(urban.items(), key=lambda item: item[1], reverse=True)}

    return set(urban.keys())

def urban_to_multiword(dict_path, english_words):

    multiwords = []
    entries = []

    for letter in string.ascii_uppercase:
        with open(Path(dict_path) / f"{letter}.data", 'r', encoding='utf-8') as f:
            entries += f.read().splitlines()

    entries = entries[100000:200000]
    for entry in tqdm(entries):
        entry = entry.replace('"', '')
        if entry.startswith('a '):
            entry = entry[2:]
        elif entry.startswith('an '):
            entry = entry[3:]

        for i in range(len(entry) - 2):
            if entry[i] == entry[i + 1] == entry[i + 2]:
                continue
        
        if len(entry.split()) > 1:
            multiwords.append(entry)
    
    multiwords_clean = []

    for entry in multiwords:
        ignore = False
        for word in entry.split():
            if word not in english_words:
                ignore = True
                break

        if not ignore:
            multiwords_clean.append(entry)

    return multiwords_clean


def lemmas(dict_path):
    with open(dict_path, 'r', encoding = 'utf-8') as f:
        rows = f.read().splitlines()
    lemma_words = {}
    for row in rows:
        if row.startswith(";"):
            continue
        lemma , word = row.split(" -> ")

        if '/' in lemma:
            lemma, freq = lemma.split('/')
        else:
            lemma, freq = lemma, 1

        lemma = lemma.replace('-','')

        if len(lemma) < 3:
            continue

        word = word.replace("-","")
        words = word.split(",")

        for w in words:
            lemma_words[w] = lemma

    return lemma_words

def gpt_borrow_words(word):

        url = "https://api.openai.com/v1/completions"
        api_key = ""
        headers = {"Authorization": f"Bearer {api_key}"}

        data = {'model': 'text-davinci-003', 
                'prompt': f"In one word, what language is the word '{word}'?",
                "temperature": 0.7, 
                "max_tokens": 3
                }

        response = requests.post(url, headers=headers, json=data).json()

def seidel(path):

    with open(path, 'r', encoding='utf-8') as f:
        entries = f.read().splitlines()

    clean_entries = []
    for entry in entries:
        entry = entry.lower().strip()
        # if "(" in entry and ")" in entry:
        #     bracketed = entry[entry.find("(")+1:entry.find(")")].strip()
        #     entry = entry[:entry.find("(")].strip()
            # if bracketed[0] == '-':
            #     clean_entries.append(bracketed[1:])
            # else:
            #     clean_entries.append(bracketed)

        if "/" in entry:
            clean_entries.append(entry[:entry.find("/")].strip())
            clean_entries.append(entry[entry.find("/")+1:].strip())
            continue

        clean_entries.append(entry)

    multiwords = []
    words = []
    for entry in clean_entries:
        if len(entry.split()) > 1:
            multiwords.append(entry)
        else:
            if len(entry) > 2:
                words.append(entry)

    return words

def names(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().splitlines()