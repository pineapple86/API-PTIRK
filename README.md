# IRKPT
- Code for ``Identifying Relevant API Knowledge\\from API tutorial and Stack Overflow with Prompt Tuning``
- Please contact 1024010437@njupt.edu.cn for questions and suggestions.

## Quick Start

### Download Pretrained Models
Dynamic Prompt Generator ([BERT-Based Classifier Model](https://huggingface.co/bert-base-uncased))

Joint Extractor ([T5-Based Extractor Model](https://drive.google.com/file/d/15OFkWw8kJA1k2g_zehZ0pxcjTABY2iF1/view))

Create 'hf_models' and put all models to 'hf_models/' for training.

### Model Training

Training model as follows:

1.Run `classifier/my_classifier.py` to train Generator

2.Run `main.py` to train Extractor
