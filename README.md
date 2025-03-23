# IRKPT
- Code for ``Identifying Relevant API Knowledge from API tutorial and Stack Overflow with Prompt Tuning``
- Please contact 3199895801@qq.com for questions and suggestions.

## Quick Start

### Download Pretrained Models

#### Text model

BERT Model ([BERT](https://huggingface.co/bert-base-uncased)) or StencentBERT Model([StencentBERT](https://huggingface.co/sentence-transformers/all-mpnet-base-v2))

Create 'pre_models_text' and put all text models to 'pre_models_text' for training.

#### Code model

CodeBERT Model([CodeBERT](https://huggingface.co/microsoft/codebert-base)) or CodeT5 Model([CodeT5](https://huggingface.co/Salesforce/codet5-small))

Create 'pre_models_code' and put all code models to 'pre_models_code' for training.

### Download Datasets

For AK-McGill, TU-McGill([TU-McGill](http://docs.oracle.com/javase/tutorial/)) from ``Discovering Information Explaining API Types Using Text Classification``, and SO-McGill([SO-McGill](https://zenodo.org/records/6944137#.YuVEFurP1Jw)) from ``Retrieving API Knowledge from Tutorials and Stack Overflow Based on Natural Language Queries``

For AK-Android, TU-Android([TU-Android](http://oscar-lab.org/paper/API/)) from ``A More Accurate Model for Finding Tutorial Segments Explaining APIs``, SO-Android([SO-Android](https://zenodo.org/records/6944137#.YuVEFurP1Jw)) from ``Retrieving API Knowledge from Tutorials and Stack Overflow Based on Natural Language Queries``

### Model Training

Training model as follows:

1.Run `classifier/my_classifier.py` to train Generator

2.Run `main.py` to train Extractor
