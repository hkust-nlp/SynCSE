## Contrastive Learning of Sentence Embeddings from Scratch
This repository contains the code, datasets and pre-trained models for our paper [Contrastive Learning of Sentence embeddings from scratch](https://arxiv.org/abs/2305.15077).


**************************** **Updates** ****************************

Thanks for your interest in our repo!

* 2/6: We released our [model checkpoints and dataset](https://huggingface.co/sjtu-lit)
* 5/23: We released [our paper](https://arxiv.org/abs/2305.15077). Check it out!


## Quick Links

  - [Overview](#overview)
  - [Model List](#model-list)
  - [Train SynCSE](#train-SynCSE)
    - [Requirements](#requirements)
    - [Evaluation](#evaluation)
    - [Training](#training)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)

## Overview

Contrastive learning has been the dominant approach to train state-of-the-art sentence embeddings. Previous studies have typically learned sentence embeddings either through the use of human-annotated natural language inference (NLI) data or via large-scale unlabeled sentences in an unsupervised manner. However, even in the case of unlabeled data, their acquisition presents challenges in certain domains due to various reasons. To address these issues, we present SynCSE, a contrastive learning framework that trains sentence embeddings with synthesized data. Specifically, we explore utilizing large language models to synthesize the required data samples for contrastive learning, including (1) producing positive and negative annotations given unlabeled sentences (SynCSE-partial), and (2) generating sentences along with their corresponding annotations from scratch (SynCSE-scratch). Experimental results on sentence similarity and reranking tasks indicate that both SynCSE-partial and SynCSE-scratch greatly outperform unsupervised baselines, and SynCSE-partial even achieves comparable performance to the supervised models in most settings.





## Model List

Our released models are listed as following.  
|              Model              | Avg. STS |
|:-------------------------------|:--------:|
| [sjtu-lit/SynCSE-partial-RoBERTa-base](https://huggingface.co/sjtu-lit/SynCSE-partial-RoBERTa-base) |   81.84 |
| [sjtu-lit/SynCSE-partial-RoBERTa-large](https://huggingface.co/sjtu-lit/SynCSE-partial-RoBERTa-large) | 82.66 |
| [sjtu-lit/SynCSE-scratch-RoBERTa-base](https://huggingface.co/sjtu-lit/SynCSE-scratch-RoBERTa-base)  | 80.66 |
| [sjtu-lit/SynCSE-partial-RoBERTa-large](https://huggingface.co/sjtu-lit/SynCSE-partial-RoBERTa-large) |81.84|


The results slightly differ from what we report in the paper, due to varying data cleaning procedures. In the paper, we employ an amount of uncleaned data identical to the NLI dataset used by SimCSE. However, here we omit some unsuccessfully generated data, for example, outputs from ChatGPT might include: "I am sorry, I can not generate a paraphrased sentence of the input, because...". Specifically, for the SynCSE-partial-NLI dataset, we directly remove those sentences that fail to generate, utilizing only the remaining 263k entries for training (the source NLI dataset contains 276k entries). For the SynCSE-scratch-NLI dataset, because it is possible to produce unlabeled data, we randomly sample 276k entries.

## Dataset List
|             Dataset             | 
|:-------------------------------|
| [sjtu-lit/SynCSE-partial-NLI](https://huggingface.co/datasets/sjtu-lit/SynCSE-partial-NLI) |
| [sjtu-lit/SynCSE-scratch-NLI](https://huggingface.co/datasets/sjtu-lit/SynCSE-scratch-NLI) |

These two datasets are respectively used for the SynCSE-partial and SynCSE-scratch experimental setups. For SynCSE-partial, we use the unlabeled data from the NLI dataset used by SimCSE and generate labels for them. For SynCSE-scratch, we generate unlabeled data and their corresponding labels. After generating the data, we remove those samples that ChatGPT fails to produce successfully.

## Getting Started

### Requirements

First, install PyTorch by following the instructions from [the official website](https://pytorch.org). To faithfully reproduce our results, please use the correct `1.13.0+cu116` version corresponding to your platforms/CUDA versions. We train our model on a single A100-80G card.

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```


Then run the following script to install the remaining dependencies,

```bash
pip install -r requirements.txt
```

Then you can use our model for **encoding sentences into embeddings**
```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("sjtu-lit/SynCSE-partial-RoBERTa-large")
model = AutoModel.from_pretrained("sjtu-lit/SynCSE-partial-RoBERTa-large")
embeddings = model.encode("A woman is reading.")
```
**Compute the cosine similarities** between two groups of sentences
```python
sentences_a = ['A woman is reading.', 'A man is playing a guitar.']
sentences_b = ['He plays guitar.', 'A woman is making a photo.']
similarities = model.similarity(sentences_a, sentences_b)
```

Or build index for a group of sentences and **search** among them
```python
sentences = ['A woman is reading.', 'A man is playing a guitar.']
model.build_index(sentences)
results = model.search("He plays guitar.")
```
If you encounter any problem when directly loading the models by HuggingFace's API, you can also download the models manually from the above table and use `model = AutoModel.from_pretrained({PATH TO THE DOWNLOAD MODEL})`.

## Train SynCSE

In the following section, we describe how to train a SynCSE model by using our code.



### Evaluation
Our evaluation code for sentence embeddings is based on a modified version of [SentEval](https://github.com/facebookresearch/SentEval). It evaluates sentence embeddings on semantic textual similarity (STS) tasks and downstream transfer tasks. For STS tasks, our evaluation takes the "all" setting, and report Spearman's correlation.

Before evaluation, please download the evaluation datasets by running
```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```

Then come back to the root directory, you can evaluate any `transformers`-based pre-trained models using our evaluation code. For example,
```
bash ./scripts/eval.sh
```
which is expected to output the results in a tabular format:
```
------ test ------
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| 76.14 | 84.41 | 79.23 | 84.85 | 82.87 |    83.95     |      81.41      | 81.84 |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
```

Arguments for the evaluation script are as follows,

* `--model_name_or_path`: The name or path of a `transformers`-based pre-trained checkpoint. You can directly use the models in the above table, e.g., `princeton-nlp/sup-simcse-bert-base-uncased`.
* `--pooler`: Pooling method. Now we support
    * `cls` (default): Use the representation of `[CLS]` token.
    * `avg`: Average embeddings of the last layer. If you use checkpoints of SBERT/SRoBERTa ([paper](https://arxiv.org/abs/1908.10084)), you should use this option.
    * `avg_top2`: Average embeddings of the last two layers.
    * `avg_first_last`: Average embeddings of the first and last layers. If you use vanilla BERT or RoBERTa, this works the best.
* `--mode`: Evaluation mode
    * `test` (default): The default test mode. To faithfully reproduce our results, you should use this option.
    * `dev`: Report the development set results. Note that in STS tasks, only `STS-B` and `SICK-R` have development sets, so we only report their numbers. It also takes a fast mode for transfer tasks, so the running time is much shorter than the `test` mode (though numbers are slightly lower).
    * `fasttest`: It is the same as `test`, but with a fast mode so the running time is much shorter, but the reported numbers may be lower (only for transfer tasks).
* `--task_set`: What set of tasks to evaluate on (if set, it will override `--tasks`)
    * `sts` (default): Evaluate on STS tasks, including `STS 12~16`, `STS-B` and `SICK-R`. This is the most commonly-used set of tasks to evaluate the quality of sentence embeddings.
    * `transfer`: Evaluate on transfer tasks.
    * `full`: Evaluate on both STS and transfer tasks.
    * `na`: Manually set tasks by `--tasks`.
* `--tasks`: Specify which dataset(s) to evaluate on. Will be overridden if `--task_set` is not `na`. See the code for a full list of tasks.

### Training

**Data**

For SynCSE-partial, we generate postive and hard negative samples for each of the premise sentence in the NLI dataset used in SimCSE. For SynCSE-scratch, we generate both the unlabeled sentences and their postive and hard negative examples. Please download the dataset ([SynCSE-partial-NLI](https://huggingface.co/datasets/sjtu-lit/SynCSE-partial-NLI), [SynCSE-scratch-NLI](https://huggingface.co/datasets/sjtu-lit/SynCSE-scratch-NLI)). and put it into the data file.

**Training scripts**

We provide example training scripts for both training SynCSE in `sup_train_mp.sh`.
* `--model_name_or_path`: Pre-trained checkpoints to start with. For now we support BERT-based models (`bert-base-uncased`, `bert-large-uncased`, etc.) and RoBERTa-based models (`RoBERTa-base`, `RoBERTa-large`, etc.).
* `--temp`: Temperature for the contrastive loss.
* `--pooler_type`: Pooling method. It's the same as the `--pooler_type` in the [evaluation part](#evaluation).
* `--hard_negative_weight`: If using hard negatives (i.e., there are 3 columns in the training file), this is the logarithm of the weight. For example, if the weight is 1, then this argument should be set as 0 (default value).
* `--do_mlm`: Whether to use the MLM auxiliary objective. If True:
  * `--mlm_weight`: Weight for the MLM objective.
  * `--mlm_probability`: Masking rate for the MLM objective.

All the other arguments are standard Huggingface's `transformers` training arguments. Some of the often-used arguments are: `--output_dir`, `--learning_rate`, `--per_device_train_batch_size`.

For results in the paper, we use Nvidia A100 (80G) GPUs with CUDA 11.6 Using different types of devices or different versions of CUDA/other softwares may lead to slightly different performance.

**Hyperparameters**

We use the following hyperparamters for training SynCSE:
- Batch size: 512
- Learning rate (base): 5e-5
- Learning rate (large): 1e-5

**Convert models**

Our saved checkpoints are slightly different from Huggingface's pre-trained checkpoints. Run `python simcse_to_huggingface.py --path {PATH_TO_CHECKPOINT_FOLDER}` to convert it.



## Bugs or questions?

If you have any questions related to the code or the paper, feel free to email Junlei (`zhangjunlei@westlake.edu.cn`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation

Please cite our paper if you use SimCSE in your work:

```bibtex
@article{zhang2023contrastive,
  title={Contrastive Learning of Sentence Embeddings from Scratch},
  author={Zhang, Junlei and Lan, Zhenzhong and He, Junxian},
  journal={arXiv preprint arXiv:2305.15077},
  year={2023}
}
```
