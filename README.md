# IRKPT
- Code for ``Identifying Relevant API Knowledge\\from API tutorial and Stack Overflow with Prompt Tuning``
- Please contact 1024010437@njupt.edu.cn for questions and suggestions.

## Quick Start

### Download Pretrained Models

#### Text model

BERT Model ([BERT](https://huggingface.co/bert-base-uncased)) or StencentBERT Model([StencentBERT](https://huggingface.co/sentence-transformers/all-mpnet-base-v2))

#### Code model

CodeBERT Model([CodeBERT](https://huggingface.co/microsoft/codebert-base)) or CodeT5 Model([CodeT5](https://huggingface.co/Salesforce/codet5-small))


Create 'hf_models' and put all models to 'hf_models/' for training.

### Model Training

Training model as follows:

1.Run `classifier/my_classifier.py` to train Generator

2.Run `main.py` to train Extractor
