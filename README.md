# NArabizi corpus

This dataset is described in the paper ["*The interplay between language similarity and script on a novel multi-layer Algerian dialect corpus*"](https://arxiv.org/abs/2105.07400) by Samia Touileb and Jeremy Barnes, accepted at Findings of ACL: ACL2021.

This corpus is built on top of the NArabizi treebank by [*Seddah et all., 2020*](https://www.aclweb.org/anthology/2020.acl-main.107/) freely available ["*here*"](https://parsiti.github.io/NArabizi/). 

# Format and pre-processing

The extentions to the treebank are added in the *.conllu* files, split into pre-defined train, dev, and test sets (as inherited from the original NArabizi treebank). 

The sentiment and topic annotations are presented as one sentence per line, in the following format *conllu_ID annotation*.


# Cite

If you use this dataset or code, please cite the following paper:

```
@misc{touileb2021interplay,
      title={The interplay between language similarity and script on a novel multi-layer Algerian dialect corpus}, 
      author={Samia Touileb and Jeremy Barnes},
      year={2021},
      eprint={2105.07400},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
