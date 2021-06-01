import numpy as np
import pandas as pd
import glob
import os
import tensorflow as tf
import transformers
from transformers import TFBertForTokenClassification, TFXLMRobertaForTokenClassification
from tqdm.notebook import tqdm
import IPython

import sys
sys.path.append("..")
from data_preparation.data_preparation_pos import MBERT_Tokenizer, XLMR_Tokenizer, bert_convert_examples_to_tf_dataset, read_conll
import utils.utils as utils
import utils.pos_utils as pos_utils
import fine_tuning
import data_preparation.data_preparation_pos as data_preparation_pos
import utils.model_utils as model_utils

def test(training_lang, test_lang, split="test"):
    data_path = "../data/ud/"
    #short_model_name = "mbert"
    short_model_name = "xlm-roberta"
    experiment = "acl"
    task = "pos"
    checkpoints_path = "checkpoints/"
    trainer = fine_tuning.Trainer(training_lang, data_path, task, short_model_name)
    # Model parameters
    max_length = 256
    eval_batch_size = 8
    learning_rate = 2e-5
    epochs = 30
    tagset = pos_utils.get_ud_tags()
    num_labels = len(tagset)
    # Model creation
    trainer.build_model(max_length, batch_size, learning_rate, epochs, num_labels, tagset=tagset, eval_batch_size=64)
    weights_path = "checkpoints/" + training_lang + "/"
    weights_filename = model_name + "_pos.hdf5"
    print("Using weights from", weights_path + weights_filename)
    trainer.model.load_weights(weights_path + weights_filename)
    # Checkpoint for best model weights
    #trainer.setup_checkpoint(checkpoints_path)
    trainer.prepare_data()
    #
    test_lang_path = data_path + test_lang
    test_data, test_dataset = data_preparation_pos.load_dataset(test_lang_path, trainer.tokenizer, max_length, short_model_name, tagset=trainer.tagset, dataset_name=split)
    trainer.setup_eval(test_data, split)
    test_dataset, test_batches = model_utils.make_batches(test_dataset, eval_batch_size, repetitions=1, shuffle=False)
    #return trainer, test_data, test_dataset, test_batches
    test_preds = trainer.handle_oom(trainer.model.predict,
                                    test_dataset,
                                    steps=test_batches,
                                    verbose=1)
    test_score = trainer.metric(test_preds, test_data, split) * 100
    print("{0}-{1} {2}: {3:.1f}".format(training_lang, test_lang, split, test_score))
    return test_score

def train(checkpoint=None):
    #data_path = input("Data path: ")
    #short_model_name = input("Short model name: ")
    #experiment = input("Experiment: ")
    #task = input("Task: ")
    data_path = "../data/ud/"
    short_model_name = "xlm-roberta"
    experiment = "acl"
    task = "pos"
    checkpoints_path = "checkpoints/"
    #
    training_lang = fine_tuning.get_global_training_state(data_path, short_model_name, experiment, checkpoints_path)
    trainer = fine_tuning.Trainer(training_lang, data_path, task, short_model_name)
    #
    # Model parameters
    max_length = 256
    batch_size = 8
    learning_rate = 2e-5
    epochs = 30
    tagset = pos_utils.get_ud_tags()
    num_labels = len(tagset)
    #
    # Model creation
    trainer.build_model(max_length, batch_size, learning_rate, epochs, num_labels, tagset=tagset, eval_batch_size=64)
    #
    # Checkpoint for best model weights
    trainer.setup_checkpoint(checkpoints_path)
    #
    trainer.prepare_data()
    #
    print("Train examples:", len(trainer.train_data))
    #
    # Print an example sentence for sanity
    example_batch = trainer.train_dataset.as_numpy_iterator().next()
    for token, label in zip(example_batch[0]["input_ids"][0], example_batch[1][0]):
        if not token:
            break
        elif token == example_batch[0]["input_ids"][0][10]:
            print("...")
            break
        print("{:<25}{:<20}".format(trainer.tokenizer.decode(int(token)), tagset[label]))
    #
    try:
        trainer.setup_training()
        trainer.train()
        trainer.make_definitive()
    except KeyboardInterrupt:
        pass
    return trainer

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

    data_dir = "../data/ud/"
    results_path = "results/mbert/results_pos.xlsx"
    #results_path = "results/xlm_roberta/results_pos.xlsx"
    #full_model_name = "jplu/tf-xlm-roberta-base"
    #full_model_name = "bert-base-multilingual-cased"
    full_model_name = "tf-xlm-roberta-base"
    #short_model_name = "mbert"
    short_model_name = "xlm-roberta"
    model_name = full_model_name.split("/")[-1] # The "jplu/" part is problematic

    code_dicts = utils.make_lang_code_dicts()
    code_to_name = code_dicts["code_to_name"]
    name_to_code = code_dicts["name_to_code"]

    # Model parameters
    max_length = 256
    batch_size = 256
    tagset = ["O", "_", "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
              "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
    num_labels = len(tagset)
    label_map = {label: i for i, label in enumerate(tagset)}

    # Train models
    #train()

    # Model creation and loading weights
    # if model_name.startswith("bert"):
    #     tokenizer = MBERT_Tokenizer.from_pretrained(full_model_name)
    #     config = transformers.BertConfig.from_pretrained(full_model_name, num_labels=num_labels)
    #     model = TFBertForTokenClassification.from_pretrained(full_model_name,
    #                                                          config=config)
    # else:
    #     tokenizer = XLMR_Tokenizer.from_pretrained(full_model_name)
    #     model = TFXLMRobertaForTokenClassification.from_pretrained(full_model_name, num_labels=num_labels)

pos_eval = []


#for training_lang in ["aii", "ajp", "akk", "am", "ar", "fa", "mt", "uig", "ur"]:
#train_codes = ["fa", "uig", "ur", "he", "akk", "mt", "ar"]
#train_codes = ["ar_na", "ar_dz", "fa", "ur", "he", "mt", "ar"]
train_codes = ["fa", "ur", "he", "mt", "ar"]
for training_lang in train_codes:
        # Load and preprocess
    dev_score = test(training_lang, training_lang, "dev")
    test_score = test(training_lang, training_lang, "test")
    ar_na_score =  test(training_lang, "ar_na")
    ar_dz_score = test(training_lang, "ar_dz")

    pos_eval.append((training_lang,
                     dev_score,
                     test_score,
                     ar_na_score,
                     ar_dz_score))

pos_eval = np.array(pos_eval, dtype=object)

train_langs = [code_to_name[l] for l in train_codes]
dev_accs = pos_eval[:,1]
test_accs = pos_eval[:,2]
narabizi_accs = pos_eval[:,3]
algerian_accs = pos_eval[:,4]


table = pd.DataFrame({"Train Lang": train_langs,
                      "Dev Acc": dev_accs,
                      "Test Acc": test_accs,
                      "Narabizi": narabizi_accs,
                      "Algerian": algerian_accs
                      })

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
