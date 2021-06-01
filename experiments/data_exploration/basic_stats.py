import numpy as np
import pandas as pd
import os
import itertools
from collections import Counter
from transformers import BertTokenizer

import sys
sys.path.append("..")
import utils.utils as utils
from data_preparation.data_preparation_pos import read_conll

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

def pos_stats(info, table, tokenizer):
    file_path = info["file_path"]
    lang_name = info["lang_name"]
    dataset = info["dataset"]
    d = {}

    conllu_data = read_conll(file_path)
    examples = [{"id": sent_id, "tokens": tokens, "tags": tags} for sent_id, tokens, tags in zip(conllu_data[0], conllu_data[1], conllu_data[2])]

    # Number of examples in dataset
    if table["language"].isna().all() or lang_name not in table["language"].values:
        d["language"] = lang_name
        index = table.index[table["language"].isna()][0]
    else:
        index = table.index[table["language"] == lang_name][0]
    d[dataset + "_examples"] = [len(examples)]

    # Avg tokens
    tokens, lengths = [], []
    for e in examples:
        #print(e["tokens"])
        raw_tokens = " ".join(e["tokens"])
        tokenized = tokenizer.tokenize(raw_tokens)
        #print(tokenized)
        tokens.extend(tokenized)
        lengths.append(len(tokenized))
    d[dataset + "_avg_tokens"] = [np.array(lengths).mean()]

    # Hapaxes
    counts = np.array(list(Counter(tokens).items()))
    hapaxes = counts[counts[:,1] == "1"][:,0]
    d[dataset + "_hapaxes"] = [len(hapaxes)]
    d[dataset + "_hapaxes(%)"] = [len(hapaxes) / len(tokens) * 100]

    # Unknown
    unk = (np.array(tokens) == "[UNK]").sum()
    d[dataset + "_unknown"] = [unk]
    d[dataset + "_unknown(%)"] = [unk / len(tokens) * 100]

    table.update(pd.DataFrame(d, index=[index]))
    return table


def sentiment_stats(info, table, tokenizer):
    file_path = info["file_path"]
    lang_name = info["lang_name"]
    dataset = info["dataset"]
    d = {}

    data = pd.read_csv(file_path, header=None)
    data.columns = ["sentiment", "review"]

    # Number of examples in dataset
    if table["language"].isna().all() or lang_name not in table["language"].values:
        d["language"] = lang_name
        index = table.index[table["language"].isna()][0]
    else:
        index = table.index[table["language"] == lang_name][0]
    d[dataset + "_examples"] = [data.shape[0]]

    # Avg tokens
    tokens, lengths = [], []
    for e in data["review"]:
        tokenized = tokenizer.encode(e)
        tokens.extend(tokenized)
        lengths.append(len(tokenized))
    d[dataset + "_avg_tokens"] = [np.array(lengths).mean()]

    # Hapaxes
    counts = np.array(list(Counter(tokens).items()))
    hapaxes = counts[counts[:,1] == 1][:,0]
    d[dataset + "_hapaxes"] = [len(hapaxes)]
    d[dataset + "_hapaxes(%)"] = [len(hapaxes) / len(tokens) * 100]

    # Unknown
    unk = (np.array(tokens) == 100).sum()
    d[dataset + "_unknown"] = [unk]
    d[dataset + "_unknown(%)"] = [unk / len(tokens) * 100]

    table.update(pd.DataFrame(d, index=[index]))
    return table

if __name__ == "__main__":

    model_name = "bert-base-multilingual-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    names = ["train", "dev", "test"]
    names_examples = (np.array(names, dtype=object) + "_examples").tolist()
    names_avg = (np.array(names, dtype=object) + "_avg_tokens").tolist()
    names_hapaxes = np.array(list(itertools.product(names, ["_hapaxes", "_hapaxes(%)"])), dtype=object)
    names_hapaxes = (names_hapaxes[:,0] + names_hapaxes[:,1]).tolist()
    names_unknown = np.array(list(itertools.product(names, ["_unknown", "_unknown(%)"])), dtype=object)
    names_unknown = (names_unknown[:,0] + names_unknown[:,1]).tolist()
    colnames = ["language"] + names_examples + names_avg + names_hapaxes + names_unknown
    values = np.empty((len(os.listdir("../data/ud/")), len(colnames)))
    values[:] = np.nan

    pos_table = utils.run_through_data("../data/ud/",
                                       pos_stats,
                                       pd.DataFrame(values, columns=colnames),
                                       tokenizer=tokenizer)

    pos_table = utils.order_table(pos_table)
    pos_table = pos_table.astype(dict.fromkeys([col for col in pos_table.columns[1:] if "%" not in col and "avg" not in col],
                                           pd.Int64Dtype())) # Convert to int


    colnames_percentage = []
    colnames_avg = []

    for col in pos_table.columns:
        if "%" in col:
            colnames_percentage.append(col)
        elif "avg" in col:
            colnames_avg.append(col)

    pos_table[colnames_percentage] = pos_table[colnames_percentage].applymap(lambda x: "{:.2f}".format(x))
    pos_table[colnames_avg] = pos_table[colnames_avg].applymap(lambda x: "{:.1f}".format(x))
    pos_table = pos_table.replace(np.nan, "-")
    pos_table = pos_table.replace("nan", "-")

    print(utils.convert_table_to_latex(pos_table.iloc[:,:7]))

    print(utils.convert_table_to_latex(pos_table[["language"] + pos_table.columns[7:13].tolist()]))

    print(utils.convert_table_to_latex(pos_table[["language"] + pos_table.columns[13:].tolist()]))


    names = ["train", "dev", "test"]
    names_examples = (np.array(names, dtype=object) + "_examples").tolist()
    names_avg = (np.array(names, dtype=object) + "_avg_tokens").tolist()
    names_hapaxes = np.array(list(itertools.product(names, ["_hapaxes", "_hapaxes(%)"])), dtype=object)
    names_hapaxes = (names_hapaxes[:,0] + names_hapaxes[:,1]).tolist()
    names_unknown = np.array(list(itertools.product(names, ["_unknown", "_unknown(%)"])), dtype=object)
    names_unknown = (names_unknown[:,0] + names_unknown[:,1]).tolist()
    colnames = ["language"] + names_examples + names_avg + names_hapaxes + names_unknown
    values = np.empty((len(os.listdir("../data/sentiment/")), len(colnames)))
    values[:] = np.nan

    sentiment_table = utils.run_through_data("../data/sentiment/",
                                       sentiment_stats,
                                       pd.DataFrame(values, columns=colnames),
                                       tokenizer=tokenizer)

    sentiment_table = utils.order_table(sentiment_table, "sentiment_table.txt")
    sentiment_table = sentiment_table.astype(
    dict.fromkeys([col for col in sentiment_table.columns[1:] if "%" not in col and "avg" not in col],
                  pd.Int64Dtype())) # Convert to int

    sentiment_table.to_excel("sentiment_basic_stats.xlsx", index=False)

    colnames_percentage = []
    colnames_avg = []

    for col in sentiment_table.columns:
        if "%" in col:
            colnames_percentage.append(col)
        elif "avg" in col:
            colnames_avg.append(col)

    sentiment_table[colnames_percentage] = sentiment_table[colnames_percentage].applymap(lambda x: "{:.2f}".format(x))
    sentiment_table[colnames_avg] = sentiment_table[colnames_avg].applymap(lambda x: "{:.1f}".format(x))
    sentiment_table = sentiment_table.replace(np.nan, "-")
    sentiment_table = sentiment_table.replace("nan", "-")
    sentiment_table

    print(utils.convert_table_to_latex(sentiment_table.iloc[:,:7], "sentiment_table.txt"))

    print(utils.convert_table_to_latex(sentiment_table[["language"] + sentiment_table.columns[7:13].tolist()], "sentiment_table.txt"))
