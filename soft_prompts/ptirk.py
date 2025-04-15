import pandas as pd
import torch
import torch.nn as nn
import random
from openprompt.data_utils import InputExample
from openprompt.prompts import SoftTemplate, ManualVerbalizer
from torch.optim import AdamW
from sklearn.model_selection import KFold
from bs4 import BeautifulSoup

def parse_html_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    plain_text = soup.get_text() or ""
    code_snippets = [tt.get_text() for tt in soup.find_all('tt')] + [pre.get_text() for pre in soup.find_all('pre')]
    code_text = " ".join(code_snippets) if code_snippets else ""
    return plain_text, code_text

def read_answers(filename, sheet_name="your_sheet"):
    df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
    answers = []
    for index, row in df.iterrows():
        html_content = f"{row[2]}"
        api = f"{row[4]}"
        target = row[6]
        text_a_plain, text_a_code = parse_html_content(html_content)
        text_a_for_codebert = text_a_code
        text_b_for_codebert = api
        example_codebert = InputExample(guid=target, text_a=text_a_for_codebert, text_b=text_b_for_codebert)
        text_a_for_bert = text_a_plain
        text_b_for_bert = api
        example_bert = InputExample(guid=target, text_a=text_a_for_bert, text_b=text_b_for_bert)
        answers.append((example_codebert, example_bert))
    return answers

def shuffle_train_dataset(train_dataset):
    random.shuffle(train_dataset)
    return train_dataset

dataset = read_answers('dataset/your_data')

# Load other pre-trained models (CodeBERT/CodeT5/BERT/SentenceBERT)
from openprompt.plms import load_plm
code_plm, code_tokenizer, code_model_config, code_WrapperClass = load_plm(
    "pre_models/your_code_model"
)
text_plm, text_tokenizer, text_model_config, text_WrapperClass = load_plm(
    "pre_models/your_text_model"
)

code_promptTemplate = SoftTemplate(
    model=code_plm,
    tokenizer=code_tokenizer,
    text='{"soft":10} {"placeholder":"text_a"} {"placeholder":"text_b"} {"mask"}'
)
text_promptTemplate = SoftTemplate(
    model=text_plm,
    tokenizer=text_tokenizer,
    text='{"soft":10} {"placeholder":"text_a"} {"placeholder":"text_b"} {"mask"}'
)

code_promptVerbalizer = ManualVerbalizer(
    classes=['0', '1'],
    label_words={
        '0': ['no', 'incorrect', 'irrelevant'],
        '1': ['yes', 'correct', 'relevant'],
    },
    tokenizer=code_tokenizer,
)
text_promptVerbalizer = ManualVerbalizer(
    classes=['0', '1'],
    label_words={
        '0': ['no', 'incorrect', 'irrelevant'],
        '1': ['yes', 'correct', 'relevant'],
    },
    tokenizer=text_tokenizer,
)

from openprompt import PromptForClassification

code_promptModel = PromptForClassification(
    template=code_promptTemplate,
    plm=code_plm,
    verbalizer=code_promptVerbalizer,
)
text_promptModel = PromptForClassification(
    template=text_promptTemplate,
    plm=text_plm,
    verbalizer=text_promptVerbalizer,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
code_promptModel.to(device)
text_promptModel.to(device)

from openprompt import PromptDataLoader

def create_dataloaders(train_dataset, valid_dataset):
    train_data_loader_code = PromptDataLoader(
        dataset=[ex[0] for ex in train_dataset],
        tokenizer=code_tokenizer,
        template=code_promptTemplate,
        tokenizer_wrapper_class=code_WrapperClass,
        batch_size=8,
        shuffle=False,
        max_seq_length=492,
        num_workers=0
    )
    valid_data_loader_code = PromptDataLoader(
        dataset=[ex[0] for ex in valid_dataset],
        tokenizer=code_tokenizer,
        template=code_promptTemplate,
        tokenizer_wrapper_class=code_WrapperClass,
        batch_size=8,
        shuffle=False,
        max_seq_length=492,
        num_workers=0
    )
    train_data_loader_text = PromptDataLoader(
        dataset=[ex[1] for ex in train_dataset],
        tokenizer=text_tokenizer,
        template=text_promptTemplate,
        tokenizer_wrapper_class=text_WrapperClass,
        batch_size=8,
        shuffle=False,
        max_seq_length=492,
        num_workers=0
    )
    valid_data_loader_text = PromptDataLoader(
        dataset=[ex[1] for ex in valid_dataset],
        tokenizer=text_tokenizer,
        template=text_promptTemplate,
        tokenizer_wrapper_class=text_WrapperClass,
        batch_size=8,
        shuffle=False,
        max_seq_length=492,
        num_workers=0
    )
    return train_data_loader_code, valid_data_loader_code, train_data_loader_text, valid_data_loader_text

class GatingFusion(nn.Module):
    def __init__(self, logits_dim):
        super(GatingFusion, self).__init__()
        self.gate = nn.Linear(logits_dim * 2, logits_dim)

    def forward(self, logits_code, logits_text):
        combined_input = torch.cat((logits_code, logits_text), dim=-1)
        gate_values = torch.sigmoid(self.gate(combined_input))
        combined_logits = gate_values * logits_code + (1 - gate_values) * logits_text
        return combined_logits

def train_kfold(code_model, text_model, dataset, num_folds=10):
    fusion_model = GatingFusion(logits_dim=2).to(device)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    for fold_idx, (train_index, valid_index) in enumerate(kf.split(dataset)):
        train_dataset = [dataset[i] for i in train_index]
        valid_dataset = [dataset[i] for i in valid_index]

        train_data_loader_code, valid_data_loader_code, train_data_loader_text, valid_data_loader_text = create_dataloaders(
            train_dataset, valid_dataset
        )
        max_epochs = 10
        lr = 2e-5
        adam_epsilon = 1e-8
        optimizer_codebert = AdamW([
            {'params': code_promptTemplate.parameters(), 'lr': lr},
            {'params': [p for p in code_plm.parameters() if p.requires_grad], 'lr': lr}
        ], eps=adam_epsilon)

        optimizer_bert = AdamW([
            {'params': text_promptTemplate.parameters(), 'lr': lr},
            {'params': [p for p in text_plm.parameters() if p.requires_grad], 'lr': lr}
        ], eps=adam_epsilon)
        optimizer_fusion = AdamW(fusion_model.parameters(), lr=lr)
        for epoch_idx in range(max_epochs):
            train_dataset = shuffle_train_dataset(train_dataset)
            train_data_loader_code, _, train_data_loader_text, _ = create_dataloaders(
                train_dataset, valid_dataset
            )
            total_loss = 0.0
            num_batches = 0
            code_model.train()
            text_model.train()
            fusion_model.train()
            for batch_code, batch_text in zip(train_data_loader_code, train_data_loader_text):
                batch_code.to(device)
                batch_text.to(device)
                labels = batch_code['guid'].to(device)
                logits_code = code_model(batch_code)
                logits_text = text_model(batch_text)
                combined_logits = fusion_model(logits_code, logits_text)
                loss = nn.CrossEntropyLoss()(combined_logits, labels)
                total_loss += loss.item()
                loss.backward()
                optimizer_codebert.step()
                optimizer_bert.step()
                optimizer_fusion.step()
                optimizer_codebert.zero_grad()
                optimizer_bert.zero_grad()
                optimizer_fusion.zero_grad()
                num_batches += 1

train_kfold(code_promptModel, text_promptModel, dataset)
