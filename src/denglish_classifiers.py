import itertools
import os
import pickle
import re
import string
import time
import warnings

import emoji
import numpy as np
import torch
import utils.classifier_feature_util as clfutil
from nltk import ngrams
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn_crfsuite import CRF
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          BertTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments)
from utils.corpus import Corpus

from tokenizations import get_alignments

os.environ['TOKENIZERS_PARALLELISM'] = 'false' 

#### BEGIN K-FOLD CROSS-VALIDATION ####################################################################################

def k_fold_cross_validation(X, y, clf, k=10, shuffle=False):
    """Perform k-fold cross-validation on classifier clf using input samples X and targets y.

    Returns a tuple of:
      - a list of batches of test samples (list of NumPy-arrays)
      - a list of batches of targets (list of NumPy-arrays)
      - a list of batches of predictions (list of NumPy-arrays)
    sorted by rounds of cross-validation (i.e. the first element of each list is a NumPy-array containing the samples/
    /targets/predictions from the first round).
    X and y must be iterable, and clf must have methods fit and predict, such that clf.fit(X, y) and clf.predict(X)
    train the classifier or classify samples respectively.
    Pass shuffle=True to shuffle samples before splitting.
    """
    # Convert X and y to NumPy-arrays to make sure we can index them with arrays. (Note that converting to list first
    # may be necessary, e.g. to collect the elements produced by a generator instead of the generator itself.)
    
    X = np.array(list(X), dtype=object)
    y = np.array(list(y), dtype=object)

    sample_list, target_list, pred_list = [], [], []
    kf = KFold(n_splits=k, shuffle=shuffle)
    for train_idxs, test_idxs in kf.split(X):
        print("CROSS-VALI-ITER")
        np.random.shuffle(train_idxs)
        X_train = X[train_idxs]
        y_train = y[train_idxs]
        X_test = X[test_idxs]
        y_test = y[test_idxs]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        sample_list.append(X_test.copy())
        target_list.append(y_test.copy())
        pred_list.append(y_pred.copy())

    return sample_list, target_list, pred_list

def mbert_k_fold_cross_validation(all_subword_labels, all_word_labels, all_word_tokens, pretrained_name, k=10, shuffle=False):

    print(pretrained_name)
    sample_list, target_list, pred_list = [], [], []
    
    id2label = {0: 'D', 1: 'M', 2: 'E', 3: 'O', 4: 'SE', 5: 'SD', 6: 'SO'}
    label2id = {'D': 0, 'M': 1, 'E': 2, 'O': 3, 'SE': 4, 'SD': 5, 'SO': 6}

    kf = KFold(n_splits=k, shuffle=shuffle)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    for train_idxs, test_idxs in kf.split(all_subword_labels):
        X_train = []
        y_train = []
        word_tokens_train = []
        word_labels_train = []

        X_test = []
        y_test = []
        word_tokens_test = []
        word_labels_test = []
        
        for i, idx in enumerate(train_idxs):
            X_train.append(all_word_tokens[idx])
            y_train.append(all_subword_labels[idx])
            word_tokens_train.append(all_word_tokens[idx])
            word_labels_train.append(all_word_labels[idx])

        for i, idx in enumerate(test_idxs):
            X_test.append(all_word_tokens[idx])
            y_test.append(all_subword_labels[idx])
            word_tokens_test.append(all_word_tokens[idx])
            word_labels_test.append(all_word_labels[idx])

            assert len(all_word_labels[idx]) == len(all_word_tokens[idx])

        # Initialize classifier and perform k-fold cross-validation.
        model = AutoModelForTokenClassification.from_pretrained(pretrained_name, num_labels=7, id2label=id2label, label2id=label2id)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

        # Call the data loading and preprocessing function
        data_collator, train_dataset = load_and_preprocess_data(
            X_train, y_train, tokenizer, label2id
        )

        training_args = TrainingArguments(
            output_dir="/home/is473/rds/hpc-work/Denglisch/corpus/mBERT_model",
            per_device_train_batch_size=16,  
            save_strategy='no',
            num_train_epochs=3, 
            learning_rate=3e-5,
            weight_decay=0.01
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            tokenizer=tokenizer
        )

        # Start training
        trainer.train()

        predictions = []
        trues = []
            
        for word_tokens, true_labels in zip(word_tokens_test, word_labels_test):

            inputs = tokenizer(word_tokens, is_split_into_words=True, return_tensors="pt", truncation=True).to(device)

            subword_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

            with torch.no_grad():
                outputs = model(**inputs)

            # Get the predicted labels
            predicted_subword_labels = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
            
            assert len(subword_tokens) == len(predicted_subword_labels), (word_tokens, word_labels, true_labels)

            # Convert predicted labels to your mapping
            predicted_subword_labels = [id2label[label] for label in predicted_subword_labels]

            word_labels = get_subword_labels(subword_tokens, word_tokens, predicted_subword_labels)

            assert len(word_labels) == len(true_labels), (word_tokens, word_labels, true_labels)

            predictions.append(word_labels)
            trues.append(true_labels)

        sample_list.append(predictions)
        pred_list.append(predictions)
        target_list.append(trues)

    return sample_list, target_list, pred_list

def get_subword_labels(a, b, a_labels):
    a2b, b2a = get_alignments(a, b)

    # Assign labels to subwords
    b_labels = []
    most_common = 'D'

    for i, label_indices in enumerate(b2a):

        aligned_subwords = []

        if label_indices:
            for j in label_indices:
                if j < len(a_labels):
                    aligned_subwords.append(a_labels[j])

        if not aligned_subwords:
            aligned_subwords = [most_common]

        most_common = max(set(aligned_subwords), key=aligned_subwords.count)

        b_labels.append(most_common)
    
    return b_labels

def load_and_preprocess_data(train_tokens, train_subword_labels, tokenizer, label2id):

    train_encodings = tokenizer(train_tokens, is_split_into_words=True)
    train_encodings = {'input_ids': train_encodings['input_ids'], 'labels': train_subword_labels}

    train_encodings['labels'] = [[label2id[t] for t in sent] for sent in train_encodings['labels']]

    data_collator = DataCollatorForTokenClassification(tokenizer)

    return data_collator, ClassificationDataset(train_encodings)

class ClassificationDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])


#### END K-FOLD CROSS-VALIDATION ######################################################################################


#### BEGIN CRF-SPECIFIC STUFF #########################################################################################

# Function to count the occurrences of "D" in an inner list
def count_D(inner_list):
    return sum(1 for elem in inner_list if elem == "D")

def print_crf_metrics(label_list, sample_list, target_list, pred_list, name):
    print("Printing CRF METRICS")

    filtered_indices = []
    for i, lists in tqdm(enumerate(target_list)):
        for j, inner_list in enumerate(lists):
            d_count = count_D(inner_list)
            if len(inner_list) > 0 and d_count / len(inner_list) >= 0.5 and ("E" in inner_list or "M" in inner_list):
                filtered_indices.append((i, j))

    pred_list = [[[inner_list[k] for k in range(len(inner_list)) if (i, j) in filtered_indices] for j, inner_list in enumerate(lists)] for i, lists in enumerate(pred_list)]
    target_list = [[[inner_list[k] for k in range(len(inner_list)) if (i, j) in filtered_indices] for j, inner_list in enumerate(lists)] for i, lists in enumerate(target_list)]

    indices_to_keep = [(i, j, k) for i, lists in enumerate(target_list) for j, inner_list in enumerate(lists) for k, label in enumerate(inner_list) if label not in ['O', 'SO']]

    pred_list = [[[inner_list[k] for k in range(len(inner_list)) if (i, j, k) in indices_to_keep] for j, inner_list in enumerate(lists)] for i, lists in enumerate(pred_list)]
    pred_list = [[[('E' if elem == 'SE' else 'D' if elem in ['SD', 'SO', 'O'] else elem) for elem in inner_list] for inner_list in lists] for lists in pred_list]

    target_list = [[[inner_list[k] for k in range(len(inner_list)) if (i, j, k) in indices_to_keep] for j, inner_list in enumerate(lists)] for i, lists in enumerate(target_list)]
    target_list = [[[('E' if elem == 'SE' else 'D' if elem in ['SD', 'SO'] else elem) for elem in inner_list] for inner_list in lists] for lists in target_list]

    target_list = [np.array(l, dtype=object) for l in target_list]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        num_of_cats = len(label_list)
        k = len(sample_list)

        tag_id_dictionary = dict()
        for index in range(len(label_list)):
            tag = label_list[index]
            tag_id_dictionary[tag] = index

        support=np.array([0]*num_of_cats)
        precision_score_arr,recall_score_arr=np.array([0.0]*num_of_cats),np.array([0.0]*num_of_cats)
        acc_score_arr_sen,precision_score_arr_sen,recall_score_arr_sen=np.array([0.0]*num_of_cats),np.array([0.0]*num_of_cats),np.array([0.0]*num_of_cats)
        f1_macro,precision_macro,recall_macro=0.0,0.0,0.0
        f1_weighted,precision_weighted,recall_weighted=0.0,0.0,0.0
        f1_micro,precision_micro,recall_micro=0.0,0.0,0.0
        acc_score=0.0
        acc_score_sen_level=0.0
        total_supp=0

        for X_test, y_test, y_pred in zip(sample_list, target_list, pred_list):
            y_test_f = []
            for sent_sublist in y_test.tolist():
                for tag_val in sent_sublist:
                    y_test_f.append(tag_val)
            y_pred_f = []
            for sent_sublist in y_pred:
                for tag_val in sent_sublist:
                    y_pred_f.append(tag_val)

            p, r, f1, s = metrics.precision_recall_fscore_support(y_test_f,y_pred_f,labels=label_list,average=None)
            for i in range(0,num_of_cats) :
                precision_score_arr[i]+=p[i]
                recall_score_arr[i]+=r[i]
                support[i]+=s[i]

            acc_score+=metrics.accuracy_score(y_test_f,y_pred_f)
            p, r, f1, _ = metrics.precision_recall_fscore_support(y_test_f,y_pred_f,average='macro')
            precision_macro+=p
            recall_macro+=r
            p, r, f1, _ = metrics.precision_recall_fscore_support(y_test_f,y_pred_f,average='micro')
            precision_micro+=p
            recall_micro+=r
            p, r, f1, _ = metrics.precision_recall_fscore_support(y_test_f,y_pred_f,average='weighted')
            precision_weighted+=p
            recall_weighted+=r
            total_supp=sum(support)
            '''CRF when converting tags to binary (sentence level):'''
            y_pred_sen_level= []
            test_y_sents_sen_level=[]

            for tags in y_pred:
                binary_tags=[0]*num_of_cats
                for tag in tags:
                    binary_tags[tag_id_dictionary[tag]]=1
                y_pred_sen_level.append(binary_tags)
            for tags in y_test:
                binary_tags=[0]*num_of_cats
                for tag in tags:
                    binary_tags[tag_id_dictionary[tag]]=1
                test_y_sents_sen_level.append(binary_tags)

            for i in range(0, num_of_cats):
                y_true=[sen[i] for sen in test_y_sents_sen_level]
                y_pred=[sen[i] for sen in y_pred_sen_level]
                precision_score_arr_sen[i]+=metrics.precision_score(y_true, y_pred)
                recall_score_arr_sen[i]+=metrics.recall_score(y_true, y_pred)
                acc_score_arr_sen[i]+=metrics.accuracy_score(y_true, y_pred)
            acc_score_sen_level+=metrics.accuracy_score(test_y_sents_sen_level,y_pred_sen_level)
        precision_macro=precision_macro/k
        recall_macro=recall_macro/k
        f1_macro=2.0*precision_macro*recall_macro/(precision_macro+recall_macro)
        precision_weighted=precision_weighted/k
        recall_weighted=recall_weighted/k
        f1_weighted=2.0*precision_weighted*recall_weighted/(precision_weighted+recall_weighted)
        precision_micro=precision_micro/k
        recall_micro=recall_micro/k
        f1_micro=2.0*precision_micro*recall_micro/(precision_micro+recall_micro)

        #pretty print
        print('{0}-fold WORD-level:'.format(k))
        print("---- {0} fold cross validation of the model----".format(k))
        print(acc_score/k)
        print('Category     precision     recall      f1-score      support')
        for i in range(0, num_of_cats):
            precision_score=100*precision_score_arr[i]/k
            recall_score=100*recall_score_arr[i]/k
            f1_score=2*precision_score*recall_score/(recall_score+precision_score)
            print(' {0}  :  {1:.2f}         {2:.2f}      {3:.2f}        {4}  '.format(label_list[i],precision_score,recall_score,f1_score,support[i]))
        print('micro avg  : {0:.2f}         {1:.2f}      {2:.2f}        {3} '.format(100*precision_micro,100*recall_micro,100*f1_micro,total_supp))
        print('macro avg  : {0:.2f}         {1:.2f}      {2:.2f}        {3} '.format(precision_macro,recall_macro,f1_macro,total_supp))
        print('weighted avg:{0:.2f}         {1:.2f}      {2:.2f}        {3} '.format(precision_weighted,recall_weighted,f1_weighted,total_supp))


        print('{0}-fold CRF indirect  sentence level:'.format(k))
        print("total acc score = {0:.3f}".format(acc_score_sen_level/k))
        print('Category     accuracy     precision    recall      f1-score')
        for i in range(0, num_of_cats):
            precision_score=precision_score_arr_sen[i]/k
            recall_score=recall_score_arr_sen[i]/k
            f1_score=2*precision_score*recall_score/(recall_score+precision_score)
            acc_score=acc_score_arr_sen[i]/k
            print('{0} :    {1:.2f}          {2:.2f}         {3:.2f}         {4:.2f} '.format(label_list[i],acc_score,precision_score,recall_score,f1_score))


def word2features(sent, i, most_freq_ngrams=[]):
    """
    :param sent: the sentence
    :param i: the index of the token in sent
    :param tags: the tags of the given sentence (sent)
    :return: the features of the token at index i in sent
    """
    word = sent[i]

    lower_word = word.lower()
    list_of_ngrams = list(ngrams(lower_word, 2)) + list(ngrams(lower_word, 3))
    list_of_ngrams = [''.join(ngram) for ngram in list_of_ngrams]


    features = {
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word_with_digit': any(char.isdigit() for char in word) and word.isnumeric() is False,
        'word_pure_digit': word.isnumeric(),
        'word_with_umlaut': any(char in "üöäÜÖÄß" for char in word),
        'word_with_punct': any(char in string.punctuation for char in word),
        'word_pure_punct': all(char in string.punctuation for char in word),
        'frequent_en_word': lower_word in clfutil.FreqLists.EN_WORD_LIST,
        'frequent_de_word': lower_word in clfutil.FreqLists.DE_WORD_LIST,
        'frequent_ngrams_de': any(ngram in clfutil.MOST_COMMON_NGRAMS_DE for ngram in list_of_ngrams),
        'frequent_ngrams_en': any(ngram in clfutil.MOST_COMMON_NGRAMS_EN for ngram in list_of_ngrams),
        'is_in_emoticonlist': lower_word in clfutil.OtherLists.EMOTICON_LIST,
        'is_emoji': any(char in emoji.EMOJI_DATA for char in word),

        #derivation and flextion
        'D_Der_A_suff': any(lower_word.endswith(silbe) for silbe in list(itertools.chain.from_iterable(clfutil.FlexDeri.D_DER_A_suf_dict.values()))),
        'D_Der_N_suff': any(lower_word.endswith(silbe) for silbe in list(itertools.chain.from_iterable(clfutil.FlexDeri.D_DER_N_suf_dict.values()))),
        'D_Der_V_pref': any(lower_word.startswith(silbe) for silbe in clfutil.FlexDeri.D_DER_V_pref_list),
        'E_Der_A_suff': any(lower_word.endswith(silbe) for silbe in clfutil.FlexDeri.E_DER_A_suf_list),
        'E_Der_N_suff': any(lower_word.endswith(silbe) for silbe in list(itertools.chain.from_iterable(clfutil.FlexDeri.E_DER_N_suf_dict.values()))),
        'E_Der_V_pref': any(lower_word.startswith(silbe) for silbe in clfutil.FlexDeri.E_DER_V_pref_list),
        'D_Der_V_suff': any(lower_word.endswith(silbe) for silbe in list(itertools.chain.from_iterable(clfutil.FlexDeri.D_DER_V_suf_dict.values()))),
        'E_Der_V_suff': any(lower_word.endswith(silbe) for silbe in list(itertools.chain.from_iterable(clfutil.FlexDeri.E_DER_V_suf_dict.values()))),
        'D_Flex_A_suff': any(lower_word.endswith(silbe) for silbe in list(itertools.chain.from_iterable(clfutil.FlexDeri.D_FLEX_A_suf_dict.values()))),
        'D_Flex_N_suff': any(lower_word.endswith(silbe) for silbe in clfutil.FlexDeri.D_FLEX_N_suf_list),
        'D_Flex_V_suff': any(lower_word.endswith(silbe) for silbe in clfutil.FlexDeri.D_FLEX_V_suf_list),
        'E_Flex_A_suff': any(lower_word.endswith(silbe) for silbe in clfutil.FlexDeri.E_FLEX_A_suf_list),
        'E_Flex_N_suff': any(lower_word.endswith(silbe) for silbe in clfutil.FlexDeri.E_FLEX_N_suf_list),
        'E_Flex_V_suff': any(lower_word.endswith(silbe) for silbe in clfutil.FlexDeri.E_FLEX_V_suf_list),
        'D_Flex_V_circ': lower_word.startswith("ge") and (lower_word.endswith("en") or lower_word.endswith("t")),

        #NE:
        'D_NE_Demo_suff': any(lower_word.endswith(silbe) for silbe in clfutil.NELexMorph.D_NE_Demo_suff),
        'D_NE_Morph_suff': any(lower_word.endswith(silbe) for silbe in clfutil.NELexMorph.D_NE_Morph_suff),
        'E_NE_Demo_suff': any(lower_word.endswith(silbe) for silbe in clfutil.NELexMorph.E_NE_Demo_suff),
        'E_NE_Morph_suff': any(lower_word.endswith(silbe) for silbe in clfutil.NELexMorph.E_NE_Morph_suff),
        'O_NE_Morph_suff': any(lower_word.endswith(silbe) for silbe in clfutil.NELexMorph.O_NE_Morph_suff),
        'D_NE_parts': any(silbe in lower_word for silbe in clfutil.NELexMorph.D_NE_parts),
        'E_NE_parts': any(silbe in lower_word for silbe in clfutil.NELexMorph.E_NE_parts),
        'O_NE_parts': any(lower_word.endswith(silbe) for silbe in clfutil.NELexMorph.O_NE_suff),

        #entity lists
        'D_NE_REGs': any(w in lower_word for w in clfutil.NELists.D_NE_REGs)
                     or lower_word in clfutil.NELists.D_NE_REGs_abbr,
        'E_NE_REGs': any(w in lower_word for w in clfutil.NELists.E_NE_REGs)
                     or lower_word in clfutil.NELists.E_NE_REGs_abbr,
        'O_NE_REGs': any(w in lower_word for w in clfutil.NELists.O_NE_REGs)
                     or lower_word in clfutil.NELists.O_NE_REGs_abbr
                     or any(lower_word.startswith(w) for w in clfutil.NELists.O_REG_demonym_verisons),

        'D_NE_ORGs': lower_word in clfutil.NELists.D_NE_ORGs,
        'E_NE_ORGs': lower_word in clfutil.NELists.E_NE_ORGs,
        'O_NE_ORGs': lower_word in clfutil.NELists.O_NE_ORGs,

        'D_NE_VIPs': lower_word in clfutil.NELists.D_NE_VIPs,
        'E_NE_VIPs': lower_word in clfutil.NELists.E_NE_VIPs,
        'O_NE_VIPs': lower_word in clfutil.NELists.O_NE_VIPs,

        'D_NE_PRESS': lower_word in clfutil.NELists.D_NE_PRESS,
        'E_NE_PRESS': lower_word in clfutil.NELists.E_NE_PRESS,
        'O_NE_PRESS': lower_word in clfutil.NELists.O_NE_PRESS,

        'D_NE_COMPs': lower_word in clfutil.NELists.D_NE_COMPs,
        'E_NE_COMPs': lower_word in clfutil.NELists.E_NE_COMPs,
        'O_NE_COMPs': lower_word in clfutil.NELists.O_NE_COMPs,

        'NE_MEASURE': any(w in lower_word for w in clfutil.NELists.NE_MEASURE),

        'D_CULT': any(w in lower_word for w in clfutil.CultureTerms.D_CULT),
        'E_CULT': any(w in lower_word for w in clfutil.CultureTerms.E_CULT),
        'O_CULT': any(w in lower_word for w in clfutil.CultureTerms.O_CULT),

        'D_FuncWords': lower_word in clfutil.FunctionWords.deu_function_words,
        'E_FuncWords': lower_word in clfutil.FunctionWords.eng_function_words,

        'Interj_Word': lower_word in clfutil.OtherLists.Interj_Words,

        'URL': any(lower_word.startswith(affix) for affix in clfutil.OtherLists.URL_PREF) or any(lower_word.endswith(affix) for affix in clfutil.OtherLists.URL_SUFF) or any(affix in lower_word for affix in clfutil.OtherLists.URL_INFIX)
    }

    for ngram in most_freq_ngrams:
        features[ngram] = ngram in list_of_ngrams

    if i > 0:
        pass
    else:
        features['BOS'] = True

    if i == len(sent) - 1:
        features['EOS'] = True

    return features


def sent2features(sent, most_freq_ngrams=[]):
    """
    This function returns a list of features of each token in the given sentence (and using the corresponding tags)
    """
    return [word2features(sent, i, most_freq_ngrams) for i in range(len(sent))]

def get_ngrams(word_list, num_of_ngrams):
    ngrams_dict = dict()
    for word in word_list:
        ngram_list = [''.join(ngram) for ngram in list(ngrams(word, 2)) + list(ngrams(word, 3))]
        for ngram in ngram_list:
            if ngram in ngrams_dict.keys():
                ngrams_dict[ngram] += 1
            else:
                ngrams_dict[ngram] = 1
    sorted_list = sorted(ngrams_dict.items(), key=lambda item: item[1],reverse=True)

    res_lst = [strng for strng, value in sorted_list[:num_of_ngrams]]
    return res_lst

#### END CRF-SPECIFIC STUFF ###########################################################################################


#### BEGIN MAIN CODE ##################################################################################################

LABEL_LIST_FULL = ['1', '2',
                   '3', '3a', '3a-E', '3a-D', '3a-AE', '3a-AD', '3b', '3c', '3c-C', '3c-M', '3c-EC', '3c-EM',
                   '3-D', '3-E', '3-O',
                   '4', '4a', '4b', '4b-E', '4b-D', '4c', '4d', '4d-E', '4d-D', '4e-E',
                   '<punct>', '<EOS>', '<EOP>', '<url>']
LABEL_LIST_COLLAPSED = ['E', 'D', 'M', 'SE', 'SD', 'SO', 'O']
LABEL_LIST_FURTHER_COLLAPSED = ["E", "D", "M"]


def main_crf_cross_validation(file_name, label_list, num_rounds, name):

    print(f"label list: {label_list}")
    corpus = Corpus(file_name)

    # Find most frequent N-grams.
    word_list, _ = corpus.get_tokens()
    most_freq_ngrams = get_ngrams(word_list, 200)

    # Preprocess data: toks is a list containing one list of tokens for each sentence in the corpus, tags is the
    # corresponding list of lists of tags. We extract features from toks using sent2features(), the tags are already
    # the targets we want.
    toks, tags = corpus.get_sentences()

    toks = [t for t in toks if not len(t) > 100]
    tags = [t for t in tags if not len(t) > 100]

    print("start features")
    X = [sent2features(s, most_freq_ngrams) for s in toks]
    y = tags

    # Initialize classifier and perform 10-fold cross-validation.
    print("start classifier")
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
    print_crf_metrics(label_list, *k_fold_cross_validation(X, y, crf, k=num_rounds, shuffle=False), name = name)

def main_mbert_cross_validation(file_name, label_list, num_rounds, pretrained_name, name):
    
    tokenizer = BertTokenizer.from_pretrained(pretrained_name)

    print(f"label list: {label_list}")

    corpus = Corpus(file_name)

    # Load your data here
    toks, tags = corpus.get_sentences()

    toks_filtered = []
    tags_filtered = []
    all_subword_labels = []

    for sent_toks, sent_tags in zip(toks, tags):

        if len(sent_toks) > 100:
            continue

        sent_toks = [t.replace("’", "'").replace("”", "'").replace("“", "'").replace("„", "'").replace("―", "-").replace("–", "-").replace("…", "...").replace("`", "'").replace("‘", "'").replace("—", "-").replace("´", "'").replace("'¯\\ \\_(ツ)_/¯'", '!').replace("¯", "-") for t in sent_toks]

        sent_toks = replace_emojis_with_X(sent_toks)

        subword_ids = tokenizer(sent_toks, is_split_into_words=True)
        subword_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in subword_ids["input_ids"]]

        subword_labels = get_subword_labels(sent_toks, subword_tokens, sent_tags)

        assert len(subword_labels) == len(subword_tokens)

        all_subword_labels.append(subword_labels)
        tags_filtered.append(sent_tags)
        toks_filtered.append(sent_toks)

    print_crf_metrics(label_list, *mbert_k_fold_cross_validation(all_subword_labels, tags_filtered, toks_filtered, pretrained_name, k=num_rounds, shuffle=False), name=name)

def replace_emojis_with_X(tokens):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return ['X' if re.match(emoj, token) else token for token in tokens]

def main_rules_test(target_file, pred_file, label_list, name):

    target_corpus = Corpus(target_file)
    pred_corpus = Corpus(pred_file)

    target_toks, target_list = target_corpus.get_sentences()

    target_toks = [t for t in target_toks if not len(t) > 100]
    target_list = [t for t in target_list if not len(t) > 100]

    pred_toks, pred_list = pred_corpus.get_sentences()
    pred_toks = [t for t in pred_toks if not len(t) > 100]
    pred_list = [t for t in pred_list if not len(t) > 100]

    print_crf_metrics(label_list, [target_list], [target_list], [pred_list], name=name)


def main():

    print("Denglish CRF")
    main_crf_cross_validation("../data/denglish/Manu_corpus_collapsed.csv", LABEL_LIST_FURTHER_COLLAPSED, 10, name='denglish') # 10-fold cross-validation

    print("tsBERT")
    main_mbert_cross_validation("../data/denglish/Manu_corpus_collapsed.csv", LABEL_LIST_FURTHER_COLLAPSED, 10, pretrained_name='igorsterner/german-english-code-switching-bert', name='tsbert') 

    print("TongueSwitcher")
    main_rules_test("../data/denglish/Manu_corpus_collapsed.csv", "../data/resources/denglish_labelled_with_tongueswitcher.csv", LABEL_LIST_FURTHER_COLLAPSED, name='tongueswitcher') # 10-fold cross-validation

#### END MAIN CODE ####################################################################################################


if __name__ == "__main__":
    main()
