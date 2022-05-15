import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent.absolute().__str__()
sys.path.append(BASE_DIR)
import os
import gc
import re
import ast
import random
import itertools
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.models.deberta_v2 import DebertaV2TokenizerFast
from config import *
from src import dataset, utils

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIR = os.path.join(BASE_DIR, 'input')
LOGGER = utils.get_logger()
utils.seed_everything(seed=42)

if DEBUG:
    test = pd.read_csv(os.path.join(INPUT_DIR, 'nbme-score-clinical-patient-notes', 'train.csv'))[:300]
    test = test[['id', 'case_num', 'pn_num', 'feature_num']]
    LOGGER.info('DEBUGGING MODE ON, USING TRAINING DATA')
else:
    test = pd.read_csv(os.path.join(INPUT_DIR, 'nbme-score-clinical-patient-notes', 'test.csv'))
submission = pd.read_csv(os.path.join(INPUT_DIR, 'nbme-score-clinical-patient-notes', 'sample_submission.csv'))
features = pd.read_csv(os.path.join(INPUT_DIR, 'nbme-score-clinical-patient-notes', 'features.csv'))
def preprocess_features(features):
    features.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"
    return features
features = preprocess_features(features)
patient_notes = pd.read_csv(os.path.join(INPUT_DIR, 'nbme-score-clinical-patient-notes', 'patient_notes.csv'))
test = test.merge(features, on=['feature_num', 'case_num'], how='left')
test = test.merge(patient_notes, on=['pn_num', 'case_num'], how='left')

def inference_fn_fast(test_loader, model, max_len, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        bs = len(inputs['input_ids'])
        pred_w_pad = np.zeros((bs, max_len, 1))
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        y_preds = y_preds.sigmoid().to('cpu').numpy()
        pred_w_pad[:, :y_preds.shape[1]] = y_preds
        preds.append(pred_w_pad)
    predictions = np.concatenate(preds)
    return predictions


def sort_df_by_token_len(test, tokenizer, max_len, bs):
    input_lengths = []
    tk0 = tqdm(zip(test['pn_history'].fillna("").values, test['feature_text'].fillna("").values), total=len(test))
    for text, feature_text in tk0:
        length = len(tokenizer(text, feature_text, add_special_tokens=True)['input_ids'])
        if m == 'google-electra-large-discriminator' and length > max_len:
            input_lengths.append(max_len)
        else:
            input_lengths.append(length)
    test['input_lengths'] = input_lengths
    length_sorted_idx = np.argsort([-len_ for len_ in input_lengths])

    # sort dataframe
    sort_df = test.iloc[length_sorted_idx]

    # calc max_len per batch
    sorted_input_length = sort_df['input_lengths'].values
    batch_max_length = np.zeros_like(sorted_input_length)

    for i in range((len(sorted_input_length)//bs)+1):
        batch_max_length[i*bs:(i+1)*bs] = np.max(sorted_input_length[i*bs:(i+1)*bs])    
    sort_df['batch_max_length'] = batch_max_length
    
    return sort_df, length_sorted_idx


class CustomModel(nn.Module):
    def __init__(self, model=None, fc_dropout=0.2, config_path=None, pretrained=False):
        super().__init__()
        if config_path is None:
            self.config = AutoConfig.from_pretrained(model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        return last_hidden_states

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output


all_predictions = {}
for m in MODELS.keys():
    print(m)
    
    if 'pl' in m:
        model_name = m.replace('-pl', '')
    elif 'retrained' in m:
        model_name = m.replace('-retrained', '')
    elif m == 'public-deberta-large':
        model_name = 'microsoft-deberta-large'
    else:
        model_name = m
    tokenizer_path = os.path.join(INPUT_DIR, MODELS_PATH[m], 'tokenizer')

    if model_name == 'microsoft-deberta-v3-large' or model_name == 'microsoft-deberta-v2-xlarge':
        tokenizer = DebertaV2TokenizerFast.from_pretrained(tokenizer_path)
    elif model_name == 'roberta-large':
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trim_offsets=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
    truncate = True if m == 'google-electra-large-discriminator' else False
    tokenizer_len = MODELS_LEN[m]
    batch_size = MODELS_BATCH_SIZE[m]
    
    sort_df, length_sorted_idx = sort_df_by_token_len(test, tokenizer, tokenizer_len, batch_size)
    
    test_dataset = dataset.TestDatasetFast(tokenizer, sort_df, truncate)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
    
    predictions = []
    for fold in TRN_FOLD:
        if m == 'public-deberta-large' and fold == 4:
            pass
        else:
            config_path = os.path.join(INPUT_DIR, MODELS_PATH[m], 'config.pth')
            model_path = os.path.join(INPUT_DIR, MODELS_PATH[m], f"{model_name}_fold{fold}_best.pth")
            model = CustomModel(config_path=config_path, pretrained=False)
            state = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state['model'])
            prediction = inference_fn_fast(test_loader, model, tokenizer_len, device)
            prediction = prediction.reshape((len(test), tokenizer_len))
            prediction = prediction[np.argsort(length_sorted_idx)] 
            char_probs = utils.get_char_probs(test['pn_history'].values, prediction, tokenizer)
            predictions.append(char_probs)
            del model, state, prediction, char_probs; gc.collect()
            torch.cuda.empty_cache()
        
    predictions = np.mean(predictions, axis=0)
    all_predictions[m] = predictions

del predictions; gc.collect()


avg_predictions = None
for m, wt in MODELS.items():
    if avg_predictions is None:
        avg_predictions = all_predictions[m] * wt
    else:
        avg_predictions = avg_predictions + all_predictions[m] * wt

if not DEBUG:
    results = utils.get_results(avg_predictions, test['pn_history'].values, th=BEST_TH)
    submission['location'] = results
    submission[['id', 'location']].to_csv('submission.csv', index=False)