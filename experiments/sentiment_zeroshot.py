import numpy as np
import pandas as pd
import glob
import os
import tensorflow as tf
import transformers
from transformers import (TFBertForSequenceClassification, BertTokenizer,
                         TFXLMRobertaForSequenceClassification, XLMRobertaTokenizer,
                         TFBertForTokenClassification, TFXLMRobertaForTokenClassification)
from tqdm.notebook import tqdm
import IPython

import sys
sys.path.append("..")
from data_preparation.data_preparation_pos import MBERT_Tokenizer, XLMR_Tokenizer, bert_convert_examples_to_tf_dataset, read_conll
import utils.utils as utils
import utils.pos_utils as pos_utils
import fine_tuning
import data_preparation.data_preparation_sentiment as data_preparation_sentiment
import utils.model_utils as model_utils

def test(training_lang, test_lang, split="test"):
    data_path = "../data/sentiment/"
    #short_model_name = "mbert"
    short_model_name = "xlm-roberta"
    experiment = "acl_sentiment"
    task = "sentiment"
    checkpoints_path = "checkpoints/"
    trainer = fine_tuning.Trainer(training_lang, data_path, task, short_model_name)
    # Model parameters
    max_length = 256
    eval_batch_size = 8
    learning_rate = 2e-5
    epochs = 30
    num_labels = 2
    # Model creation
    trainer.build_model(max_length, batch_size, learning_rate, epochs, num_labels, eval_batch_size=eval_batch_size)
    weights_path = "checkpoints/" + training_lang + "/"
    weights_filename = model_name + "_sentiment.hdf5"
    trainer.model.load_weights(weights_path + weights_filename)
    # Checkpoint for best model weights
    test_lang_path = data_path + test_lang
    test_data, test_dataset = data_preparation_sentiment.load_dataset(test_lang_path, trainer.tokenizer, max_length, short_model_name, dataset_name=split)
    #trainer.setup_eval(test_data, "test")
    test_dataset, test_batches = model_utils.make_batches(test_dataset, eval_batch_size, repetitions=1, shuffle=False)
    test_preds = trainer.handle_oom(trainer.model.predict,
                                    test_dataset,
                                    steps=test_batches,
                                    verbose=1)
    test_score = trainer.metric(test_preds, test_data, split) * 100
    print("{0}-{1} {2}: {3:.1f}".format(training_lang, test_lang, test_score))
    return test_score

def train(checkpoint=None):

    data_path = "../data/sentiment/"
    task = "sentiment"
    #short_model_name = "mbert"
    short_model_name = "xlm-roberta"
    experiment = "acl_sentiment"
    checkpoints_path = "checkpoints/"
    use_class_weights = False

    training_lang = fine_tuning.get_global_training_state(data_path, short_model_name, experiment, checkpoints_path)
    trainer = fine_tuning.Trainer(training_lang, data_path, task, short_model_name, use_class_weights)

    # Model parameters
    max_length = 256
    batch_size = 8
    learning_rate = 2e-5
    epochs = 20

    # Model creation
    trainer.build_model(max_length, batch_size, learning_rate, epochs, num_labels=2, eval_batch_size=32)

    # Checkpoint for best model weights
    trainer.setup_checkpoint(checkpoints_path)

    trainer.prepare_data()

    print("Train examples:", trainer.train_data.shape[0])

    print("Example review:")
    print("{0}\t{1}".format(trainer.train_data["sentiment"][0],
                            trainer.train_data["review"][0]))
    if trainer.use_class_weights:
        trainer.calc_class_weights()
        print(trainer.class_weights)

    trainer.setup_training()
    trainer.train()
    trainer.make_definitive()

def prepare_test_data(trainer, limit=None, task="pos"):
    # Load plain data and TF dataset
    data, dataset = data_preparation_pos.load_dataset(
            trainer.lang_path, trainer.tokenizer, trainer.max_length, trainer.short_model_name,
            tagset=trainer.tagset, dataset_name="test")
    trainer.setup_eval(data, "test")
    dataset, batches = model_utils.make_batches(
            dataset, trainer.eval_batch_size, repetitions=1, shuffle=False)
    return (dataset, batches, data)



def setup_eval(data, tokenizer, label_map, max_length, dataset_name="test"):
    eval_info = {}
    eval_info[dataset_name] = {}
    eval_info[dataset_name]["all_words"] = []
    eval_info[dataset_name]["all_labels"] = []
    eval_info[dataset_name]["real_tokens"] = []
    eval_info[dataset_name]["subword_locs"] = []
    acc_lengths = 0
    #
    for i in range(len(data)):
        eval_info[dataset_name]["all_words"].extend(data[i]["tokens"]) # Full words
        eval_info[dataset_name]["all_labels"].extend([label_map[label] for label in data[i]["tags"]])
        _, _, idx_map = tokenizer.subword_tokenize(data[i]["tokens"], data[i]["tags"])
        # Examples always start at a multiple of max_length
        # Where they end depends on the number of resulting subwords
        example_start = i * max_length
        example_end = example_start + len(idx_map)
        eval_info[dataset_name]["real_tokens"].extend(np.arange(example_start, example_end, dtype=int))
        # Get subword starts and ends
        sub_ids, sub_starts, sub_lengths = np.unique(idx_map, return_counts=True, return_index=True)
        sub_starts = sub_starts[sub_lengths > 1] + acc_lengths
        sub_ends = sub_starts + sub_lengths[sub_lengths > 1]
        eval_info[dataset_name]["subword_locs"].extend(np.array([sub_starts, sub_ends]).T.tolist())
        acc_lengths += len(idx_map)
    return eval_info

def get_score_pos(preds, dataset_name, eval_info):
        filtered_preds = preds[0].argmax(axis=-1).flatten()[eval_info[dataset_name]["real_tokens"]].tolist()
        filtered_logits = preds[0].reshape(
            (preds[0].shape[0]*preds[0].shape[1], preds[0].shape[2])
        )[eval_info[dataset_name]["real_tokens"]]
        new_preds = pos_utils.reconstruct_subwords(
            eval_info[dataset_name]["subword_locs"], filtered_preds, filtered_logits
        )
        return (np.array(eval_info[dataset_name]["all_labels"]) == np.array(new_preds)).mean()

if __name__ == "__main__":

    data_dir = "../data/sentiment/"
    results_path = "results/mbert/results_sentiment.xlsx"
    #results_path = "results/xlm_roberta/results_pos.xlsx"
    #full_model_name = "jplu/tf-xlm-roberta-base"
    #full_model_name = "bert-base-multilingual-cased"
    full_model_name = "xlm-roberta-base"
    #short_model_name = "mbert"
    short_model_name = "xlm-roberta"
    model_name = full_model_name.split("/")[-1] # The "jplu/" part is problematic

    code_dicts = utils.make_lang_code_dicts()
    code_to_name = code_dicts["code_to_name"]
    name_to_code = code_dicts["name_to_code"]

    # Model parameters
    max_length = 256
    batch_size = 256
    num_labels = 2

    # Train models
    #train()

    # Model creation and loading weights
    if model_name.startswith("bert"):
        tokenizer = MBERT_Tokenizer.from_pretrained(full_model_name)
        config = transformers.BertConfig.from_pretrained(full_model_name, num_labels=num_labels)
        model = TFBertForSequenceClassification.from_pretrained(full_model_name, config=config)
    else:
        tokenizer = XLMR_Tokenizer.from_pretrained(full_model_name)
        model = TFXLMRobertaForSequenceClassification.from_pretrained(full_model_name, num_labels=num_labels)

sent_eval = []



train_codes = ["ar_na", "ar_dz", "fa", "ur", "he", "mt", "ar"]
for training_lang in train_codes:
    weights_path = "checkpoints/" + training_lang + "/"
    weights_filename = model_name + "_sentiment.hdf5"
    model.load_weights(weights_path + weights_filename)
    print("Using weights from", weights_path + weights_filename)

    for test_lang in ["ar_na", "ar_dz"]:
        # Load and preprocess
        dev_score, test_score, cross_score = test(training_lang, test_lang)
        sent_eval.append((training_lang, test_lang, dev_score, test_score, cross_score))

sent_eval = np.array(sent_eval, dtype=object)

train_langs = [code_to_name[l] for l in train_codes]
dev_accs = sent_eval[:,2]
dev_accs = dev_accs[::2]
test_accs = sent_eval[:,3]
na_accs = sent_eval[:,4]
narabizi_accs = na_accs[::2]
algerian_accs = na_accs[1::2]


table = pd.DataFrame({"Train Lang": train_langs,
                      "Dev": dev_accs,
                      "Test": test_accs,
                      "Narabizi": narabizi_accs,
                      "Algerian": algerian_accs
                      })

# Change language ids to languages
#table["Train Lang"] = table["Train Lang"].apply(lambda x: code_to_name[x])
#table["Test Lang"] = table["Test Lang"].apply(lambda x: code_to_name[x])

# Multiply accuries by 100
table["Dev"] *= 100
table["Test"] *= 100
table["Narabizi"] *= 100
table["Algerian"] *= 100


#file = open("data_exploration/pos_table.txt", "r")
#lang_order = [line.split("&")[1].strip() for line in file.readlines()]
#table["sort"] = table["Language"].apply(lambda x: lang_order.index(x))
#table = table.sort_values(by=["sort"]).drop("sort", axis=1).reset_index(drop=True)

print(table)
print(table.to_latex(index=False, float_format="{0:.1f}".format))

# if os.path.isfile(results_path):
#     results = pd.read_excel(results_path, sheet_name=None)
# else:
#     results = dict.fromkeys(table.columns[1:].values, pd.DataFrame({"Language": table["Language"].values}))

# with pd.ExcelWriter(results_path) as writer:
#     full_training_lang = code_to_name[training_lang]
#     for sheet_name, df in results.items():
#         # Add each the column for each metric in the corresponding sheet
#         df[full_training_lang] = table[sheet_name]
#         df.to_excel(writer, index=False, sheet_name=sheet_name)
