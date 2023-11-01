// This is the base environment file
// It serves as default values for all other jsonnet config files
// Please override these values dirrectly in corresponding config files

local seed=2021;

// data path configuration
local wandb_cache_dir = '../cache/wandb';
local default_cache_folder = '../data/cache';
local dictionary_cache_file = '../data/cache/dictionaries.pkl';
local affixes_cache_file = '../data/cache/affixes.pkl';
local zenodo_tweets_path = '../data/zenodo_tweets';
local max_number_tweets = 1000;
local dictionaries = {
  "dictcc": {
    "dict_path": "../data/dictionaries/dictcc",
    "dicts": {
      "dictcc": {
        "file_path": "dictcc.txt",
        "lan": "de",
      },
    },
  },
  "urban": {
    "dict_path": "../data/wordlist_data/urban_dictionary",
    "dicts": {
      "urban": {
        "file_path": "",
        "lan": "en",
      },
      
    },
  },
  "leipzig": {
    "dict_path": "../data/wordlist_data/wortschatz-leipzig",
    "dicts": {
      "austrian": {
        "file_path": "100k-de-at-news-2012.txt",
        "lan": "de",
      },
      "swiss": {
        "file_path": "300k-de-ch-news-2012.txt",
        "lan": "de",
      },
      "german": {
        "file_path": "300k-de-news-2021.txt",
        "lan": "de",
      },
      "english": {
        "file_path": "100k-en-news-2020.txt",
        "lan": "en",
      },
    },
  },
};
{
  "DATA_FOLDER": "",
  "EXPERIMENT_FOLDER": "",
  "TENSORBOARD_FOLDER": "",
  "WANDB": {
    "CACHE_DIR":  wandb_cache_dir,
    "entity": "YOUR ENTITY",
    "project": "YOUR PROJECT",
    "tags": [""],
  },
  "platform_type": "",
  "ignore_pretrained_weights": [],
  "experiment_name": "default_test",
  "seed": seed,
  "cache":{
    "default_folder": default_cache_folder,
  },
  "data_loader": {
    "dataset_modules": {
      "module_list": [],
      "module_dict":{   // all available modules
        "LoadTweets": {
          "type": "DataLoaderTweets", "option": "default",
          "config": {
            "tweets_path": zenodo_tweets_path,
            "custom_tweets": 0,
            "playground_input_path": "../data/playground_input/playground.txt",
          },
        },
        "LoadDictionaries": {
          "type": "DataLoaderDictionaries", "option": "default",
          "config": {
            "dictionaries_path": dictionaries,
            "dictionary_cache": dictionary_cache_file,
            "affixes_cache": affixes_cache_file,
          },
        },
        "LoadPretrainingTweets": {
        },
        "LoadClassificationTweets": {
        },
      },
    },
  },
  "cuda": 0,
  "gpu_device":0,
}
