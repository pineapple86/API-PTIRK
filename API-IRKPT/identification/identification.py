import torch
import torch.nn as nn
from openprompt.data_utils import InputExample
from openprompt.prompts import SoftTemplate, ManualVerbalizer
from openprompt import PromptForClassification, PromptDataLoader
from bs4 import BeautifulSoup
from openprompt.plms import load_plm

def parse_html_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    plain_text = soup.get_text() or ""
    code_snippets = [tt.get_text() for tt in soup.find_all('tt')] + [pre.get_text() for pre in soup.find_all('pre')]
    code_text = " ".join(code_snippets) if code_snippets else ""
    return plain_text, code_text

def preprocess_new_data(api, knowledge_item):
    plain_text, code_text = parse_html_content(knowledge_item)
    example_codebert = InputExample(guid=None, text_a=code_text, text_b=api)
    example_bert = InputExample(guid=None, text_a=plain_text, text_b=api)
    return example_codebert, example_bert

def create_dataloader_for_new_data(example_codebert, example_bert, code_tokenizer, text_tokenizer):
    new_data_loader_code = PromptDataLoader(
        dataset=[example_codebert],
        tokenizer=code_tokenizer,
        template=code_promptTemplate,
        tokenizer_wrapper_class=code_WrapperClass,
        batch_size=1,
        shuffle=False,
        max_seq_length=490,
        num_workers=0
    )
    new_data_loader_text = PromptDataLoader(
        dataset=[example_bert],
        tokenizer=text_tokenizer,
        template=text_promptTemplate,
        tokenizer_wrapper_class=text_WrapperClass,
        batch_size=1,
        shuffle=False,
        max_seq_length=490,
        num_workers=0
    )
    return new_data_loader_code, new_data_loader_text

def predict_new_data(code_model, text_model, fusion_model, new_data_loader_code, new_data_loader_text):
    code_model.eval()
    text_model.eval()
    fusion_model.eval()
    with torch.no_grad():
        for batch_code, batch_text in zip(new_data_loader_code, new_data_loader_text):
            batch_code = batch_code.to(device)
            batch_text = batch_text.to(device)
            logits_code = code_model(batch_code)
            logits_text = text_model(batch_text)
            combined_logits = fusion_model(logits_code, logits_text)
            preds = torch.argmax(combined_logits, dim=-1).cpu().numpy()
            return preds

class GatingFusion(nn.Module):
    def __init__(self, logits_dim):
        super(GatingFusion, self).__init__()
        self.gate = nn.Linear(logits_dim * 2, logits_dim)

    def forward(self, logits_code, logits_text):
        combined_input = torch.cat((logits_code, logits_text), dim=-1)
        gate_values = torch.sigmoid(self.gate(combined_input))
        combined_logits = gate_values * logits_code + (1 - gate_values) * logits_text
        return combined_logits

if __name__ == "__main__":
    code_plm, code_tokenizer, _, code_WrapperClass = load_plm("pre_models/your_code_model")
    text_plm, text_tokenizer, _, text_WrapperClass = load_plm("pre_models/your_text_model")

    code_promptTemplate = SoftTemplate(model=code_plm, tokenizer=code_tokenizer, text='{"soft":10} {"placeholder":"text_a"} {"placeholder":"text_b"} {"mask"}')
    text_promptTemplate = SoftTemplate(model=text_plm, tokenizer=text_tokenizer, text='{"soft":10} {"placeholder":"text_a"} {"placeholder":"text_b"} {"mask"}')

    code_promptVerbalizer = ManualVerbalizer(
        classes=['0', '1'],
        label_words={
            '0': ['no', 'incorrect', 'irrelevant'],
            '1': ['yes', 'correct', 'relevant'],
        },
        tokenizer=code_tokenizer
    )
    text_promptVerbalizer = ManualVerbalizer(
        classes=['0', '1'],
        label_words={
            '0': ['no', 'incorrect', 'irrelevant'],
            '1': ['yes', 'correct', 'relevant'],
        },
        tokenizer=text_tokenizer
    )

    code_promptModel = PromptForClassification(template=code_promptTemplate, plm=code_plm, verbalizer=code_promptVerbalizer)
    text_promptModel = PromptForClassification(template=text_promptTemplate, plm=text_plm, verbalizer=text_promptVerbalizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    code_promptModel.load_state_dict(torch.load("best_model/your_best_code.pth", map_location=device))
    text_promptModel.load_state_dict(torch.load("best_model/your_best_text.pth", map_location=device))

    fusion_model = GatingFusion(logits_dim=2).to(device)
    fusion_model.load_state_dict(torch.load("best_model/your_best_fusion.pth", map_location=device))

    code_promptModel.to(device)
    text_promptModel.to(device)

    new_data = [
        {
            "api": "your_api",
            "knowledge_item": """
            your_knowledge_item
            """
        },
    ]

    processed_data = [preprocess_new_data(item["api"], item["knowledge_item"]) for item in new_data]
    data_loaders = [create_dataloader_for_new_data(code_ex, text_ex, code_tokenizer, text_tokenizer) for code_ex, text_ex in processed_data]

    predictions = [predict_new_data(code_promptModel, text_promptModel, fusion_model, loader_code, loader_text) for loader_code, loader_text in data_loaders]

    for i, pred in enumerate(predictions):
        print(f"API: {new_data[i]['api']}")
        print(f"Knowledge Item: {new_data[i]['knowledge_item']}")
        print(f"Predicted Relevance: {'Relevant' if pred == 1 else 'Irrelevant'}")
        print("-" * 50)