import json
import logging
import os

logger = logging.getLogger(__name__)

# from utils.text_cleaner import TextCleaner


class SavingProcessor:
    
    def __init__(self) -> None:
        pass

    def do_nothing_metric(self, module, data_dict, log_dict):
        return log_dict

    def write_corpus_to_file(self, data_dict, module=None, log_dict=None) -> dict:
        
        with open(os.path.join(self.config.tongueswitcher_path, f"annotation_{self.config.year}_{self.config.month}.jsonl"), 'a', encoding='utf-8') as f:
            for id in data_dict:                
                num_en_or_mixed = sum(1 for token in data_dict[id]["anno"] if (token["lan"] == 'E' or token["lan"] == 'M'))
                num_de = sum(1 for token in data_dict[id]["anno"] if token["lan"] == 'D')

                if num_en_or_mixed > 1 and num_de > len(data_dict[id]["anno"]) // 2:

                    json_line = {"text": data_dict[id]["text"], "date": data_dict[id]["date"], "annotation": data_dict[id]["anno"]}
                    json.dump(json_line, f)
                    f.write('\n')

    def write_annotation_to_file(self, data_dict, module=None, log_dict=None) -> dict:
        """
        Write detection to file in experiment folder
        """
        
        with open(os.path.join(self.config.results_path, f"annotations.jsonl"), 'a', encoding='utf-8') as f:
            for tweet_id in data_dict:
                # labels = [a["lan"] for a in data_dict[tweet_id]["anno"] if a["lan"] != 'D']
                json_line = {"tweet_id": tweet_id, "text": data_dict[tweet_id]["text"], "date": data_dict[tweet_id]["date"], "annotation": data_dict[tweet_id]["anno"]}
                json.dump(json_line, f)
                f.write('\n')