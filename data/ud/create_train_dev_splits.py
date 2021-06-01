import os

def read_conll(input_file):
        """Reads a conllu file."""
        texts = []
        #
        text = []
        for line in open(input_file, encoding="utf-8"):
            if line.startswith("#") or line[0].isnumeric():
                text.append(line)
            elif line.strip() == "":
                texts.append(text)
                text = []
        return texts

if __name__ == "__main__":

    langs = ["Amharic", "Akkadian", "Assyrian", "South Levantine Arabic"]
    codes = ["am", "akk", "aii", "ajp"]

    for lang, code in zip(langs, codes):
        files = [f for f in os.listdir(code)]
        test_file = [t for t in files if "test.conllu" in t][0]
        examples = read_conll(os.path.join(code, test_file))
        split_id = int(len(examples) * 0.9)
        train, dev = examples[:split_id], examples[split_id:]
        with open(os.path.join(code, "train.conllu"), "w") as outfile:
            for t in train:
                text = "".join(t) + "\n"
                outfile.write(text)
        with open(os.path.join(code, "dev.conllu"), "w") as outfile:
            for t in dev:
                text = "".join(t) + "\n"
                outfile.write(text)
