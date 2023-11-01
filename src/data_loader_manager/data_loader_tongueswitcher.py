import csv
import json
import os
import random
import sys

from tqdm import tqdm

from utils.cleaning import *
from utils.unpack_dictionaries import *

maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)

import logging
from pathlib import Path

from easydict import EasyDict

logger = logging.getLogger(__name__)
import pickle

from tqdm import tqdm

from data_loader_manager.data_loader_wrapper import DataLoaderWrapper


class DataLoaderCS(DataLoaderWrapper):

    def __init__(self, config):
        DataLoaderWrapper.__init__(self, config)

    def LoadTweets(self, module_config):
        """
        This function loads the input data
        {
          "type": "LoadTweets", "option": "default",
          "config": {
            "zenodo_tweets_path": ""
          },
        },
        """

        if self.config.args.playground_input:
            with open(module_config.config.playground_input_path, 'r', encoding='utf-8') as f:
                sentences = f.read().splitlines()
                sentences = {str(i): sentence for i, sentence in enumerate(sentences)}

            sentences = {id: {"date": "2023", "text": t} for id, t in sentences.items()}
            
            self.data.zenodo_tweets = [EasyDict(sentences)]

            return
            

        else:
            tweets = {}
            json_files_path = Path(module_config.config.tweets_path) / str(self.config.year) / str(self.config.month)
            for json_file in tqdm(json_files_path.iterdir()):
                if json_file.is_file() and json_file.suffix == '.json':
                    with open(json_file, 'r', encoding = 'utf-8') as f:
                        print(f"Opening: {json_file}")
                        data = json.load(f)
                else:
                    continue
                for tweet in data:
                    if "text" in tweet:
                        tweets[tweet["id"]] = {
                            "date": tweet["created_at"][:10],
                            "text": tweet["text"].replace("\n", " ")
                        }

                # Load 1.5 times to account for cleaning removal
                if "max_number_tweets" in self.config and len(tweets) > 1.5*self.config.max_number_tweets:
                    tweets = {k: tweets[k] for k in list(tweets)[:int(1.5*self.config.max_number_tweets)]}
                    print(f"Stopping extracting tweets as {len(tweets)} found!")
                    break
                
        print("Number of tweets before cleaning: ", len(tweets))

        for tool in self.config.cleaning:
            tweets = globals()[tool](tweets)

        print("Number of tweets used as input: ", len(tweets))

        if "max_number_tweets" in self.config and len(tweets) > self.config.max_number_tweets:
            tweets = dict(random.sample(list(tweets.items()), self.config.max_number_tweets))
        
        print("Cut to: ", len(tweets))

        batched_tweets = self.batch_tweets(tweets, batch_size=1000)
               
        self.data.zenodo_tweets = batched_tweets

    @staticmethod
    def batch_tweets(tweets, batch_size=1000):
        keys, values, batched_tweets = list(tweets.keys()), list(tweets.values()), []
        
        for index in range(0, len(keys), batch_size):
            batch_keys, batch_values = keys[index:index+batch_size], values[index:index+batch_size]
            batched_tweets.append(EasyDict(dict(zip(batch_keys, batch_values))))
        
        return batched_tweets

    def LoadDictionaries(self, module_config):

        if (not self.config.args.reset_dictionaries) and (os.path.isfile(module_config.config.dictionary_cache) and os.path.isfile(module_config.config.affixes_cache)):
            print("Opening dictionary and affix cache...")
            with open(module_config.config.dictionary_cache, 'rb') as f:
                self.data.dictionaries = pickle.load(f) 
            with open(module_config.config.affixes_cache, 'rb') as f:
                self.data.affixes = pickle.load(f)            
            return

        # Compiling the wordlists from scratch

        dictionaries = module_config.config.dictionaries_path
        dict_paths = []
        for dict_type in dictionaries.keys():
            dict_dir = dictionaries[dict_type]
            for dict_name in dictionaries[dict_type]["dicts"].keys():
                dict_in = dict_dir["dicts"][dict_name]
                dict_file = Path(dict_dir["dict_path"]) / dict_in["file_path"]
                dict_paths.append(
                    {
                    "type": dict_type,
                    "path": dict_file,
                    "lan": dict_in["lan"]
                    }
                    )

        languages = ["en", "de"]
        dicts = {lan: [] for lan in languages}

        logger.info(
            f"Loading dictionaries..."
        )

        for d in tqdm(dict_paths):
            if d["lan"] in dicts:
                func_name = d["type"]
                # print(f"Loading {func_name}...")
                new_words = globals()[func_name](d["path"])
                # print(f"Adding {len(new_words)} using {func_name}")
                new_words = [word.lower() for word in new_words if len(word) > 2]
                dicts[d["lan"]] += new_words
            else:
                raise Exception("This language has not been specified")

        
        dicts = {lan: set(words) for lan, words in dicts.items()}

        german_dict_small = set(leipzig(self.config.small_german_dict_data_dir, min_freq=2))

        dicts["not_cs"] = german_dict_small

        seidel_words = seidel(self.config.seidel_path)

        for word in tqdm(dicts["de"].intersection(dicts["en"])):
            if word in seidel_words:
                dicts["de"].remove(word)    


        gpt_borrow = {}
        gpt_processed = Path(self.config.gpt_processed_data_dir)

        outputs = ['german', 'english', 'other']

        for output in outputs:
            with open(gpt_processed / f'{output}.json', 'r', encoding = 'utf-8') as f:
                print(len(gpt_borrow))
                gpt_borrow.update(json.load(f))

        gpt_borrow = dictcc_homographs(self.config.dictcc_data_dir, gpt_borrow)
        
        for n in names(self.config.boys_names_data_dir):
            if n.lower() in dicts["en"]:
                c+=1
                dicts["en"].remove(n.lower())
            if n.lower() in gpt_borrow:
                del gpt_borrow[n.lower()]

        for n in names(self.config.girls_names_data_dir):
            if n.lower() in dicts["en"]:
                d += 1
                dicts["en"].remove(n.lower())
            if n.lower() in gpt_borrow:
                del gpt_borrow[n.lower()]

        for word in tqdm(dicts["de"].intersection(dicts["en"])):
            if word in gpt_borrow:
                if gpt_borrow[word] == "english":
                    dicts["de"].remove(word)
                    if word in dicts["not_cs"]:
                        dicts["not_cs"].remove(word)
                elif gpt_borrow[word] == 'mixed':
                    dicts["de"].remove(word)
                    dicts["en"].remove(word)
                    if word in dicts["not_cs"]:
                        dicts["not_cs"].remove(word)
                else:
                    dicts["en"].remove(word)

        print("Adding two letter words")

        dicts = {lan: list(words) for lan, words in dicts.items()}

        one_twos_path = Path(self.config.one_two_words_data_dir)

        with open(one_twos_path / "two_letter_english_words.txt", 'r', encoding='utf-8') as f:
            dicts["en_one_two"] = f.read().splitlines()
            dicts["en"] += dicts["en_one_two"]

        with open(one_twos_path / "two_letter_german_words.txt", 'r', encoding='utf-8') as f:
            dicts["de_one_two"] = f.read().splitlines()
            dicts["de"] += dicts["de_one_two"]

        with open(self.config.contractions_data_dir, 'r') as f:
            contractions = json.load(f)

        for key, value in contractions.items():
            dicts["en"].append(key.lower())
            dicts["en"].append(value.lower())

        dicts = {lan: set(words) for lan, words in dicts.items()}

        with open(self.config.hard_codings_data_dir, 'r', encoding = 'utf-8') as f:
            hard_coded_words = json.load(f)

        for word in hard_coded_words.keys():
            if hard_coded_words[word] == 'english':
                if word in dicts["de"]:
                    dicts["de"].remove(word)
                if word in dicts["not_cs"]:
                    dicts["not_cs"].remove(word)
                if word not in dicts["en"]:
                    dicts["en"].add(word)
            elif hard_coded_words[word] == 'german' or hard_coded_words[word] != 'mixed':
                if word in dicts["en"]:
                    dicts["en"].remove(word)
                if word not in dicts["de"]:
                    dicts["de"].add(word)
                if word not in dicts["not_cs"]:
                    dicts["not_cs"].add(word)
            elif hard_coded_words[word] == 'mixed':
                if word in dicts["en"]:
                    dicts["en"].remove(word)
                if word in dicts["de"]:
                    dicts["de"].remove(word)
                if word in dicts["not_cs"]:
                    dicts["not_cs"].remove(word)
            else:
                if word in dicts["en"]:
                    dicts["en"].remove(word)
                if word in dicts["de"]:
                    dicts["de"].remove(word)   
                if word in dicts["not_cs"]:
                    dicts["not_cs"].remove(word)                

        with open(self.config.cities_data_dir, 'r', encoding='utf-8') as f:
            cities = f.read().splitlines()

        cities = [c for c in cities if c]
        cities = [re.sub("[\(].*?[\)]", "", c).lower().strip() for c in cities]

        logger.info("Cities 1...")
        for word in tqdm(dicts["de"].intersection(dicts["en"])):
            if word in cities:
                dicts["en"].remove(word)
                dicts["de"].remove(word)
                if word in dicts["not_cs"]:
                    dicts["not_cs"].remove(word)

        dicts = {lan: list(words) for lan, words in dicts.items()}

        for word in tqdm(dicts["de"]):
            if any(char in word for char in ["ä", "ö", "ü", "ß"]):
                plain_word = word.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss").replace("ö", "o").replace("ä", "a").replace("ü", "u")
                plain_word_2 = word.replace("ß", "ss").replace("ö", "o").replace("ä", "a").replace("ü", "u")
                if plain_word not in dicts["en"]:
                    dicts["de"].append(plain_word)
                if plain_word_2 not in dicts["en"]:
                    dicts["de"].append(plain_word_2)


        # Make sure words are lowered, they actually contain some letters and the first is a letter

        dicts = {lan: set(word.lower() for word in words if re.search(r'[a-zA-Z]', word) and word[0].isalpha()) for lan, words in dicts.items()}


        dicts["cs"] = dicts["en"] - dicts["de"]
        dicts["not_cs"] = dicts["not_cs"] - dicts["en"]

        D_DER_A_suf_dict = {"ig": ["ig", "ige", "iger", "iges", "igen", "igem"],
                            "lich": ["lich", "iche", "icher", "iches", "ichen", "ichem"],
                            "sam": ["sam", "same", "samer", "sames", "samen", "samem"],
                            "haft": ["haft", "hafte", "hafter", "haftes", "haften", "haftem"],
                            "bar": ["bar", "bare", "barer", "bares", "baren", "barem"],
                            "reich": ["reich", "reiche", "reicher", "reiches", "reichen", "reichem"],
                            "arm": ["arm", "arme", "armer", "armes", "armen", "armem"],
                            "voll": ["voll", "volle", "voller", "volles", "vollen", "vollem"],
                            "los": ["los", "lose", "loser", "loses", "losen", "losem"],
                            "isch": ["isch", "ische", "ischer", "isches", "ischen", "ischem"],
                            "frei": ["frei", "freie", "freier", "freies", "freien", "freiem"]
                            }
        
        D_DER_A_suf_list = [suffix for category in D_DER_A_suf_dict.values() for suffix in category]


        D_DER_N_suf_dict = {"heit": ["heit", "heiten"], "keit": ["keit", "keiten"],
                            "schaft": ["schaft", "schaften"], "ung": ["ung", "ungen"],
                            "nis": ["nis", "nisse", "nissen"], "tum": ["tum", "tümer", "tümern"],
                            "tät": ["tät", "täten"],
                            "in": ["in", "er"]}
        
        D_DER_N_suf_list = [suffix for category in D_DER_N_suf_dict.values() for suffix in category]

        D_DER_V_pref_list = ["ab", "an", "auf", "aus", "be", "bei", "da", "dar", "durch", "ein", "ent", "er",
                            "fort", "ge", "her", "hin", "hinter", "los", "mit", "nach", "nieder", "über",
                            "um", "un", "unter", "ur", "ver", "vor", "weg", "wieder", "zer", "zu", "zurück",
                            "zusammen", "zwischen"]

        E_DER_A_suf_list = ["ly", "ful", "able", "ible", "less", "al", "ish", "like", "ic", "ical", "ically", "ious", "ous"]

        E_DER_N_suf_dict = {"ity": ["ity", "ities"], "tion": ["tion", "tions", "ion", "ions", "ation", "ations"],
                            "logy": ["logy", "logies"], "ant": ["ant", "ants"], "hood": ["hood", "hoods"],
                            "ess": ["ess", "esses"], "ness": ["ness", "nesses"], "ism": ["ism", "isms"],
                            "ment": ["ment", "ments"],
                            "ist": ["ist", "ists"], "acy": ["acy", "acies"], "a/ence": ["ance", "ence", "ances", "ences"],
                            "dom": ["dom"]}
        E_DER_N_suf_list = [suffix for category in E_DER_N_suf_dict.values() for suffix in category]


        E_DER_V_pref_list = ["a", "after", "back", "be", "by", "down", "en",
                            "em", "fore", "hind", "mid", "midi", "mini", "mis", "off",
                            "on", "out", "over", "self", "step", "twi", "un", "under",
                            "up", "with", "re"]


        D_FLEX_V_suf_list = ["en", "et", "t", "e", "st", "est", "te"]

        D_DER_V_suf_dict = {"ieren": ["ieren", "iere", "ierst", "iert"]}
        D_DER_V_suf_list = [suffix for category in D_DER_V_suf_dict.values() for suffix in category]

        D_FLEX_A_suf_dict = {"er": ["er", "ere", "erer", "eres", "eren", "erem"],
                            "ste": ["ste", "ster", "stes", "sten", "stem"],
                            "end": ["end", "ende", "ender", "endem", "enden", "endes"]}
        D_FLEX_A_suf_list = [suffix for category in D_FLEX_A_suf_dict.values() for suffix in category]

        D_FLEX_N_suf_list = ["er", "ern", "en", "e"]

        E_FLEX_V_suf_list = ["ing", "ed", "s", "es"]

        E_DER_V_suf_dict = {"ize": ["ize", "ise", "izes", "ises", "ized", "ised"], "fy": ["fy", "fies", "fied"]}
        E_DER_V_suf_list = [suffix for category in E_DER_V_suf_dict.values() for suffix in category]

        E_FLEX_A_suf_list = ["er", "est", "st", "nd", "rd", "th"]
        E_FLEX_N_suf_list = ["s"]

        self.data["affixes"] = EasyDict()

        self.data.affixes.suffixes_de = D_DER_A_suf_list + D_DER_N_suf_list + D_DER_V_suf_list + D_FLEX_V_suf_list + D_FLEX_A_suf_list + D_FLEX_N_suf_list
        self.data.affixes.suffixes_en = E_DER_N_suf_list + E_DER_V_suf_list + E_DER_A_suf_list + E_FLEX_V_suf_list + E_FLEX_A_suf_list + E_FLEX_N_suf_list
        self.data.affixes.prefixes_de = D_DER_V_pref_list + ["lieblings"]
        self.data.affixes.prefixes_en = E_DER_V_pref_list + ["pre"]

        self.data.affixes.prefixes = set(self.data.affixes.prefixes_de + self.data.affixes.prefixes_en)
        self.data.affixes.suffixes = set(self.data.affixes.suffixes_de + self.data.affixes.suffixes_en)
        self.data.affixes.prefixes_second_de = set(["zu"])
        self.data.affixes.prefixes_second_en = set(["pre"])
        self.data.affixes.de_affixes = set(self.data.affixes.suffixes_de + self.data.affixes.prefixes_de)
        self.data.affixes.en_affixes = set(self.data.affixes.suffixes_en + self.data.affixes.prefixes_en)

        self.data.affixes.pure_prefixes_en = set(self.data.affixes.prefixes_en) - set(self.data.affixes.prefixes_de)
        self.data.affixes.pure_prefixes_de = set(self.data.affixes.prefixes_de) - set(self.data.affixes.prefixes_en)

        self.data.affixes.pure_suffixes_en = set(self.data.affixes.suffixes_en) - set(self.data.affixes.suffixes_de)
        self.data.affixes.pure_suffixes_de = set(self.data.affixes.suffixes_de) - set(self.data.affixes.suffixes_en)

        dicts["cs"] -= self.data.affixes.de_affixes 
        dicts["not_cs"] -= self.data.affixes.en_affixes 

        self.data.dictionaries = EasyDict(dicts)

        with open(module_config.config.dictionary_cache, 'wb') as f:
            pickle.dump(self.data.dictionaries, f)

        with open(module_config.config.affixes_cache, 'wb') as f:
            pickle.dump(self.data.affixes, f)


