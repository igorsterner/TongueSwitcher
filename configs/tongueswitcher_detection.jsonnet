local base_env = import '../configs/base_env.jsonnet';

local seed=2021;


local override = {
  "platform_type": "pytorch",
  "experiment_name": "default",
  "seed": seed,
  "data_loader": {
    "type": "DataLoaderCS",
    "dataset_type": "CSDataset",
    "dummy_dataloader": 0,
    "additional":{
    },
    "dataset_modules": {
      "module_list": [
        "LoadTweets",
        "LoadDictionaries",
      ],
      "module_dict":{
      },
    },
  },
  "cleaning": ["cleanbasic", "cleanrem", "cleanfastt", "cleanempty"],
  "cuda": 0,
  "gpu_device":0,

  "metrics": [
    {'name': 'write_annotation_to_file'},
  ],
  "executor": {
    "type": 'TongueSwitcher',
    "function": 'tongueswitcher_detect'
  },
  "seidel_path": "../data/wordlist_data/seidel.txt",
  "crosslingual_homographs_dir": "../data/crosslingual_homographs/cl_homograph_pos_tags.json",
  "bigram_data_dir": "../data/bigrams/bigrams.json",
  "cities_data_dir": "../data/wordlist_data/cities.txt",
  "boys_names_data_dir": "../data/wordlist_data/boys_names.txt",
  "girls_names_data_dir": "../data/wordlist_data/girls_names.txt",
  "contractions_data_dir": "../data/wordlist_data/contractions.json",
  "hard_codings_data_dir": "../data/wordlist_data/hard-coding.json",
  "small_german_dict_data_dir": "../data/wordlist_data/wortschatz_leipzig/300k-de-news-2021.txt",
  "gpt_processed_data_dir": "../data/wordlist_data/gpt",
  "dictcc_data_dir": "../data/wordlist_data/dictcc.txt",
  "one_two_words_data_dir": "../data/wordlist_data"
};

std.mergePatch(base_env, override)