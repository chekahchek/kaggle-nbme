import torch
from torch.utils.data import Dataset

def prepare_input_fast(tokenizer, text, feature_text, batch_max_len, truncate):
    inputs = tokenizer(text, feature_text, 
                       add_special_tokens=True,
                       max_length=batch_max_len,
                       padding="max_length",
                       return_offsets_mapping=False,
                       truncation=truncate)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TestDatasetFast(Dataset):
    def __init__(self, tokenizer, df, truncate):
        self.feature_texts = df['feature_text'].values
        self.pn_historys = df['pn_history'].values
        self.batch_max_len = df['batch_max_length'].values
        self.tokenizer = tokenizer
        self.truncate = truncate
        
    def __len__(self):
        return len(self.feature_texts)

    def __getitem__(self, item):
        inputs = prepare_input_fast(self.tokenizer, 
                                    self.pn_historys[item], 
                                    self.feature_texts[item],
                                    self.batch_max_len[item],
                                    self.truncate)
        return inputs