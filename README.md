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

#### AK-McGill

TU-McGill([TU-McGill](http://docs.oracle.com/javase/tutorial/)) from ``Discovering Information Explaining API Types Using Text Classification`` 

SO-McGill([SO-McGill](https://zenodo.org/records/6944137#.YuVEFurP1Jw)) from ``Retrieving API Knowledge from Tutorials and Stack Overflow Based on Natural Language Queries``


#### AK-Android

TU-Android([TU-Android](http://oscar-lab.org/paper/API/)) from ``A More Accurate Model for Finding Tutorial Segments Explaining APIs``

SO-Android([SO-Android](https://zenodo.org/records/6944137#.YuVEFurP1Jw)) from ``Retrieving API Knowledge from Tutorials and Stack Overflow Based on Natural Language Queries``

### Model Training

Run `soft_prompts/irkpt.py` for training. The following are three alternative model fusion strategies available:

```python
class WeightedAverageFusion(nn.Module):
    def __init__(self, weight_code=0.5, weight_text=0.5):
        super(WeightedAverageFusion, self).__init__()
        self.weight_code = weight_code
        self.weight_text = weight_text
    def forward(self, logits_code, logits_text):
        combined_logits = self.weight_code * logits_code + self.weight_text * logits_text
        return combined_logits
```

```python
class AttentionFusion(nn.Module):
    def __init__(self, logits_dim):
        super(AttentionFusion, self).__init__()
        self.attn = nn.Linear(logits_dim * 2, 2)
    def forward(self, logits_code, logits_text):
        combined_input = torch.cat((logits_code, logits_text), dim=-1)
        attn_weights = torch.softmax(self.attn(combined_input), dim=-1)
        weight_code = attn_weights[:, 0].unsqueeze(-1)
        weight_text = attn_weights[:, 1].unsqueeze(-1)
        combined_logits = weight_code * logits_code + weight_text * logits_text
        return combined_logits
```

```python
class SplcingFusion(nn.Module):
    def __init__(self, logits_dim):
        super(ConcatenationFusion, self).__init__()
        self.fc = nn.Linear(logits_dim * 2, logits_dim)
    def forward(self, logits_code, logits_text):
        combined_input = torch.cat((logits_code, logits_text), dim=-1)
        combined_logits = self.fc(combined_input)
        return combined_logits
```

Run `hard_prompts/prompt_H.py` for hard prompts.

### User study

1.Save your trained models and create and put them in 'best_model'.

2.Run `user_study/identification.py` for identification.

