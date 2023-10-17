local base_env = import 'base_env.jsonnet';

local seed=2021;


local override = {
  "platform_type": "pytorch",
  "experiment_name": "default",
  "seed": seed,
  "data_loader": {
    "type": "DataLoaderBertClassification",
    "dataset_type": "TongueSwitcher",
    "dummy_dataloader": 0,
    "additional":{
    },
    "dataset_modules": {
      "module_list": [
        "LoadClassificationTweets",
      ],
      "module_dict":{
      },
    },
  },
  "cuda": 0,
  "gpu_device":0,

  "metrics": [
  ],
  "bert_model_checkpoint": "igorsterner/german-english-code-switching-bert",
  "denglish_corpus": "../data/denglish/Manu_corpus_collapsed.csv"
};

std.mergePatch(base_env, override)