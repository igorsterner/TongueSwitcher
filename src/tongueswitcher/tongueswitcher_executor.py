import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import wandb
from HanTa import HanoverTagger as ht
from tongueswitcher import *

logger = logging.getLogger(__name__)

from collections import Counter
from pprint import pprint

from easydict import EasyDict
from flair.data import Sentence
from flair.models import SequenceTagger
from tongueswitcher.base_executor import BaseExecutor
from tqdm import tqdm
from utils.dirs import *


class TSExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)

        self.mixed_tagger = ht.HanoverTagger('../data/hanta_models/morphmodel_geren.pgz')
        self.en_tagger = ht.HanoverTagger('../data/hanta_models/morphmodel_en.pgz')
        self.de_tagger = ht.HanoverTagger('../data/hanta_models/morphmodel_ger.pgz')

        self.flair_tagger = SequenceTagger.load("flair/upos-multi")

    def process_tweet(self, sentence, tweet_id):

        words_processed = [{"token": token.text, "lan": "U", "pos": token.get_label("upos").value} for token in sentence]
        full_anno = getattr(self, self.config.executor.function)(words_processed)
        return full_anno, sentence, tweet_id


    def build_corpus(self):
        num_batches = len(self.data_loader.data.zenodo_tweets)

        cs_count = 0

        en_words      = []
        mixed_words   = []
        unk_words     = []
        table_entries = []

        with tqdm(total=num_batches) as pbar:
            for batch in self.data_loader.data.zenodo_tweets:
                sentences = [Sentence(batch[tweet_id]["text"]) for tweet_id in batch.keys()]
                self.flair_tagger.predict(sentences, mini_batch_size=256)
                with ThreadPoolExecutor() as executor:
                    full_annos = {}
                    futures = {executor.submit(self.process_tweet, sentence, tweet_id) for sentence, tweet_id in zip(sentences, batch.keys())}
                    for future in as_completed(futures):
                        tweet_id = future.result()[2]
                        full_annos[tweet_id] = {"anno": future.result()[0], "text": future.result()[1].text, "date": batch[tweet_id]["date"]}
                
                en_words += [t["token"] for id in full_annos.keys() for t in full_annos[id]["anno"] if t["lan"] == "E"]
                mixed_words += [t["token"] for id in full_annos.keys() for t in full_annos[id]["anno"] if t["lan"] == "M"]
                unk_words += [t["token"] for id in full_annos.keys() for t in full_annos[id]["anno"] if t["lan"] == "U"]

                for tweet_id in full_annos:

                    table_entry = [
                        batch[tweet_id]["date"],
                        full_annos[tweet_id]["text"],
                        [t["lan"] for t in full_annos[tweet_id]["anno"] if t["lan"] == "E"],
                        [t["lan"] for t in full_annos[tweet_id]["anno"] if t["lan"] == "M"],
                        [t["lan"] for t in full_annos[tweet_id]["anno"] if t["lan"] == "U"]
                    ]

                    if table_entry[2] or table_entry[3]:
                        cs_count += 1

                    table_entries.append(table_entry)

                for metric in self.config.metrics:
                    getattr(self, metric["name"])(full_annos)

                pbar.update(1)

        en_words = [w.lower() for w in en_words]
        logger.info(f"Number of tweets with english words detected: {cs_count}")
        en_counter = Counter(en_words)
        en_plot = [[word, freq] for word, freq in dict(en_counter.most_common(200)).items()]
        total = sum([i[1] for i in en_plot])
        en_plot = [[i[0], i[1]/total] for i in sorted(en_plot, key=lambda x: x[0], reverse=True)]

        mixed_words = [w.lower() for w in mixed_words]
        logger.info(f"Number of mixed words detected: {len(mixed_words)}")
        clh_counter = Counter(mixed_words)
        mixed_plot = [[word, freq] for word, freq in dict(clh_counter.most_common(200)).items()]
        total = sum([i[1] for i in mixed_plot])
        mixed_plot = [[i[0], i[1]/total] for i in sorted(mixed_plot, key=lambda x: x[0], reverse=True)]

        unk_words = [w.lower() for w in unk_words]
        logger.info(f"Number of unknown words detected: {len(unk_words)}")
        unk_counter = Counter(unk_words)
        unk_plot = [[word, freq] for word, freq in dict(unk_counter.most_common(200)).items()]
        total = sum([i[1] for i in unk_plot])
        unk_plot = [[i[0], i[1]/total] for i in sorted(unk_plot, key=lambda x: x[0], reverse=True)]


        logger.info(f"Number of code-switched words detected: {len(en_words)}")

        data_to_return = {
            "en_words": en_plot,
            "unk_words": unk_plot,
            "mixed_words": mixed_plot,
            "table_entries": table_entries,
            "full_annos": full_annos,
        }

        log_dict = self.evaluate_outputs(data_to_return)
        self.logging_results(log_dict)
        
        # return data_to_return

    def evaluate_outputs(self, step_outputs):

        columns = [
            "date",
            "tweet",
            "english words",
            "mixed_words",
            "unk words"
        ]

        test_table = wandb.Table(columns=columns)

        for i, table_entry in enumerate(step_outputs["table_entries"]):
            if i < 1000:
                test_table.add_data(*table_entry)

        columns = ["english word", "frequency"]
        en_table = wandb.Table(columns=columns)
        for i, table_entry in enumerate(step_outputs["en_words"]):
            if i < 100000:
                en_table.add_data(*table_entry)

        columns = ["mixed word", "frequency"]
        mixed_table = wandb.Table(columns=columns)
        for i, table_entry in enumerate(step_outputs["mixed_words"]):
            if i < 100000:
                mixed_table.add_data(*table_entry)

        columns = ["unkown word", "frequency"]
        unk_table = wandb.Table(columns=columns)
        for i, table_entry in enumerate(step_outputs["unk_words"]):
            if i < 100000:
                unk_table.add_data(*table_entry)
        
                
        ##############################
        ##    Compute Metrics       ##
        ##############################

        log_dict = EasyDict(
            {
                "metrics": {},
                "artifacts": {},
            }
        )
        
        log_dict.artifacts.test_table = test_table
        log_dict.artifacts.en_table = en_table
        log_dict.artifacts.mixed_table = mixed_table
        log_dict.artifacts.unk_table = unk_table

        return log_dict

    def logging_results(self, log_dict, prefix="test"):

        ### Add test results to wandb / tensorboard
        metrics_to_log = EasyDict()
        wandb_artifacts_to_log = dict()
        # Refractor the column names
        for metric, value in log_dict.metrics.items():
            metrics_to_log[f"{prefix}/{metric}"] = value

        # include other artifacts / metadata

        wandb_artifacts_to_log.update(
            {
                f"English words": log_dict.artifacts[
                    "en_table"
                ]
            }
        )

        wandb_artifacts_to_log.update(
            {
                f"Mixed words": log_dict.artifacts[
                    "mixed_table"
                ]
            }
        )

        wandb_artifacts_to_log.update(
            {
                f"Unknown words": log_dict.artifacts[
                    "unk_table"
                ]
            }
        )

        wandb_artifacts_to_log.update(
            {
                f"Code-switching detection examples": log_dict.artifacts[
                    "test_table"
                ]
            }
        )
        pprint(metrics_to_log)
        pprint(wandb_artifacts_to_log)

        if self.config.args.log_prediction_tables:
            wandb.log(
                wandb_artifacts_to_log, commit=True
            )
