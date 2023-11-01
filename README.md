# TongueSwitcher: Fine-Grained Identification of German-English Code-Switching

Code for TongueSwitcher was developed from the ML framework of [Weizhe Lin](https://github.com/LinWeizheDragon/Retrieval-Augmented-Visual-Question-Answering). This framework is designed for research purposes, with flexibility for extension. It is not a framework for production.

## 🤗 HuggingFace models

German-English code-switching BERT (tsBERT): https://huggingface.co/igorsterner/german-english-code-switching-bert

tsBERT for German-English code-switching identification: https://huggingface.co/igorsterner/german-english-code-switching-identification

## Corpus

TongueSwitcher Corpus: https://zenodo.org/records/10011601

## Wordlists

All cached wordlist resources are provided to run TongueSwitcher code-switching identification, you will not need to download anything other than this repo. Easy to read wordlists as text files are available in resources/wordlists_as_text_files. 

To compile our wordlists from scratch, you will need to download the following resources:

[Wortschatz Leipzig Corpora](https://wortschatz.uni-leipzig.de/en/download), into `data/wordlist_data/wortschatz_leipzig`:
- 100k-de-at-news-2012.txt
- 300k-de-ch-news-2012.txt
- 300k-de-news-2021.txt
- 100k-en-news-2020.txt

[dict.cc](https://www.dict.cc/?s=about%3Awordlist&l=e), into `data/wordlist_data`:
- dictcc.txt

[Urban Dictionary](https://github.com/mattbierner/urban-dictionary-word-list), into `data/wordlist_data/urban_dictionary`
- A.data
- ...
- Z.data

## Zenodo input data

To recompile the corpus using the original input tweet data, download it from [here](https://zenodo.org/record/7528718) & [here](https://zenodo.org/records/7708787) and place in the monthly folders in the `data/zenodo_tweets/` directory.

## Dependencies

We used python 3.7.16. We suggest creating a virtual environment and installing dependencies using

```
pip install -r requirements.txt
```

## TongueSwitcher Playground

You may place any text into the `data/playground_input/playground.txt` file (each line is a new data point), and the TongueSwitcher system will do German--English identification. The output will be in the `Experiments` folder, under the experiment name used.

Results will also be reported to Weights and Biases (you will need to login).

```
python main.py "../configs/tongueswitcher_detection.jsonnet" \
    --mode=tongueswitcher \
    --experiment_name=playground \
    --playground_input \
    --log_prediction_tables \
    --opts max_number_tweets=100
```

If you have downloaded all the required resources, add the `--reset_dictionaries` argument to compile the wordlists from scratch.

## TongueSwitcher Corpus Compilation:

If you have loaded the tweet data from zenodo, you may compile the TongueSwitcher corpus by month of input data using the following command. Replace Y and M with each year and month.

```
python main.py '../configs/tongueswitcher_detection.jsonnet' \
    --mode=tongueswitcher \
    --experiment_name=tongueswitcher_corpus \
    --log_prediction_tables \
    --opts year=Y month=M 
```

## BERT pretraining:

```
python main.py "../configs/bert_pretraining.jsonnet" \
    --mode=bert_pretraining \
    --experiment_name=bert_pretraining \
    --opts batch_size=32 train_epochs=1 learning_rate=1e-4
```

## tsBERT classification:

```
python main.py "../configs/tsbert_classification.jsonnet" \
    --mode=tsbert_classification \
    --experiment_name=tsbert_classification \
    --opts batch_size=16 train_epochs=3 learning_rate=3e-5
```

## Evaluation

Three evaluation scrips are provided to run our evaluation:

`tongueswitcher_evaluation.py` computes token- and entity-based metrics on the TongueSwitcher testset. It also runs inference using the TongueSwitcher rule-based system on the Denglisch corpus.

`denglisch_classifiers.py` computes metrics on this output on the Denglisch Corpus. It also implements the tsBERT cross-validation setup, as well as the original Denglisch CRF.

`interlingual_homographs_evaluation.py` implements our token-based metrics for evaluation of interlingual homographs.

Note that the dependencies suggested are insufficient to run Lingua, as their system requires python >= 3.8.

## BibTeX entry and citation info

```
@inproceedings{sterner2023tongueswitcher,
  author    = {Igor Sterner and Simone Teufel},
  title     = {TongueSwitcher: Fine-Grained Identification of German-English Code-Switching},
  booktitle = {Sixth Workshop on Computational Approaches to Linguistic Code-Switching},
  publisher = {Empirical Methods in Natural Language Processing},
  year      = {2023},
}
```