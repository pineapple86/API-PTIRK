# PTIRK
- Code for ``Using Prompt Tuning to Identify Relevant API Knowledge from API Tutorial and Stack Overflow``

## Quick Start

### Download Pretrained Models

#### Apply Prompt Tuing to these models for KIT

BERT Model ([BERT](https://huggingface.co/bert-base-uncased)) or StencentBERT Model([StencentBERT](https://huggingface.co/sentence-transformers/all-mpnet-base-v2))

#### Apply Prompt Tuing to these models for KIC

CodeBERT Model([CodeBERT](https://huggingface.co/microsoft/codebert-base)) or CodeT5 Model([CodeT5](https://huggingface.co/Salesforce/codet5-small))

Create 'pre_models' and put all models to 'pre_models' for training.

### Download Datasets

#### AK-McGill

TU-McGill([TU-McGill](http://docs.oracle.com/javase/tutorial/)) from ``Discovering Information Explaining API Types Using Text Classification`` 

SO-McGill([SO-McGill](https://zenodo.org/records/6944137#.YuVEFurP1Jw)) from ``Retrieving API Knowledge from Tutorials and Stack Overflow Based on Natural Language Queries``


#### AK-Android

TU-Android([TU-Android](http://oscar-lab.org/paper/API/)) from ``A More Accurate Model for Finding Tutorial Segments Explaining APIs``

SO-Android([SO-Android](https://zenodo.org/records/6944137#.YuVEFurP1Jw)) from ``Retrieving API Knowledge from Tutorials and Stack Overflow Based on Natural Language Queries``

### Model Tuning

Run `soft_prompts/ptirk.py` for training. The following are three alternative model fusion strategies available:

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

### Identify

1. Save your trained models and create and put them in 'best_model'.

2. Run `identification/identification.py` for identification.


### User Study

The nine APIs we randomly selected for the user study and the usefulness scores scored by the participants can be found in `user study`.

### Supplementary Experiments

Additional supplementary experiments we designed for RQ1, RQ3, and RQ4 can be found in `supplementary experiments`.

#### For RQ1

We conduct 32 supplementary experiments for RQ2, that is, randomly matching pre-trained code models (CodeBERT or CodeT5), pre-trained text models (BERT or StenceBERT), and model fusion strategies (weighted average fusion strategy, attention fusion strategy, splicing fusion strategy, or gating fusion strategy) on two datasets. In terms of F-Measure, in the 32 groups of experiments, 23 groups using soft prompts outperforms the other two prompt methods, meaning 71.88% of the results can confirm this view.

#### For RQ3

We conduct 24 supplementary experiments for RQ3, that is, randomly matching the prompt methods (hard prompts, soft prompts, or mixed prompts) and model fusion
strategies (weighted average fusion strategy, attention fusion strategy, splicing fusion strategy, or gating fusion strategy) on two datasets. In terms of F-Measure, 23 of the 24 groups of experiments, combining CodeBERT with BERT outperforms the other three pre-trained model combinations, meaning 95.83% of the results can confirm this view

#### For RQ4

We conduct 24 supplementary experiments for RQ4, that is, randomly matching pre-trained code models (CodeBERT or CodeT5), pre-trained text models (BERT or StenceBERT), and prompt methods (hard prompts, soft prompts, or mixed prompts) on two datasets. In terms of F-Measure, 17 of 24 experiment groups, using gating fusion strategy outperforms the other three model fusion strategies, meaning 70.83% of the results support this view.
