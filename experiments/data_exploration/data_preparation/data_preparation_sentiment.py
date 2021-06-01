import tensorflow as tf
from transformers.data.processors.utils import InputFeatures

class Example:
    def __init__(self, text, category_index):
        self.text = text
        self.category_index = category_index
        
def convert_examples_to_tf_dataset(
    examples,
    tokenizer,
    max_length=64,
):
    """
    Loads data into a tf.data.Dataset for finetuning a given model.

    Args:
        examples: List of tuples representing the examples to be fed
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum string length

    Returns:
        a ``tf.data.Dataset`` containing the condensed features of the provided sentences
    """
    features = [] # -> will hold InputFeatures to be converted later

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text,
            add_special_tokens=True,
            max_length=max_length, # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True, # pads to the right by default
            truncation=True
        )

        # input ids = token indices in the tokenizer's internal dict
        # token_type_ids = binary mask identifying different sequences in the model
        # attention_mask = binary mask indicating the positions of padded tokens so the model does not attend to them

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.category_index
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )

def make_batches(dataset, batch_size, repetitions, shuffle=True):
    n_batches = len(list(iter(dataset.shuffle(100000).batch(batch_size))))
    if shuffle:
        output = dataset.shuffle(100000, reshuffle_each_iteration=True).batch(batch_size).repeat(repetitions)
    else:
        output = dataset.batch(batch_size).repeat(repetitions)
    return output, n_batches