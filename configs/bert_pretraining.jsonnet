local base_env = import 'base_env.jsonnet';

local seed=2021;

local override = {
  "platform_type": "pytorch",
  "experiment_name": "default",
  "seed": seed,
  "data_loader": {
    "type": "DataLoaderBertPretrain",
    "dataset_type": "TongueSwitcher",
    "dummy_dataloader": 0,
    "additional":{
    },
    "dataset_modules": {
      "module_list": [
        "LoadPretrainingTweets",
      ],
      "module_dict":{
      },
    },
  },
  "cuda": 0,
  "gpu_device":0,

  "metrics": [
  ],
  "bert_model_checkpoint": "bert-base-multilingual-cased",
  "tongueswitcher_corpus": "../tongueswitcher-corpus",
};

std.mergePatch(base_env, override)