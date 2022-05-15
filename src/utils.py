import ast
import os
import random
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch

def micro_f1(preds, truths):
    """
    Micro f1 on binary arrays.

    Args:
        preds (list of lists of ints): Predictions.
        truths (list of lists of ints): Ground truths.

    Returns:
        float: f1 score.
    """
    # Micro : aggregating over all instances
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    return f1_score(truths, preds)


def spans_to_binary(spans, length=None):
    """
    Converts spans to a binary array indicating whether each character is in the span.

    Args:
        spans (list of lists of two ints): Spans.

    Returns:
        np array [length]: Binarized spans.
    """
    length = np.max(spans) if length is None else length
    binary = np.zeros(length)
    for start, end in spans:
        binary[start:end] = 1
    return binary


def span_micro_f1(preds, truths):
    """
    Micro f1 on spans.

    Args:
        preds (list of lists of two ints): Prediction spans.
        truths (list of lists of two ints): Ground truth spans.

    Returns:
        float: f1 score.
    """
    bin_preds = []
    bin_truths = []
    for pred, truth in zip(preds, truths):
        if not len(pred) and not len(truth):
            continue
        length = max(np.max(pred) if len(pred) else 0, np.max(truth) if len(truth) else 0)
        bin_preds.append(spans_to_binary(pred, length))
        bin_truths.append(spans_to_binary(truth, length))
    return micro_f1(bin_preds, bin_truths)

def create_labels_for_scoring(df):
    """
    Format the ground truth location such that:
    [696 724] ==> [[696, 724]] 
    [70 91, 176 183] ==> [[70, 91], [176, 193]]
    """
    df['location_for_create_labels'] = [ast.literal_eval(f'[]')] * len(df)
    for i in range(len(df)):
        lst = df.loc[i, 'location']
        if lst:
            new_lst = ';'.join(lst)
            df.loc[i, 'location_for_create_labels'] = ast.literal_eval(f'[["{new_lst}"]]')
    # create labels
    truths = []
    for location_list in df['location_for_create_labels'].values:
        truth = []
        if len(location_list) > 0:
            location = location_list[0]
            for loc in [s.split() for s in location.split(';')]:
                start, end = int(loc[0]), int(loc[1])
                truth.append([start, end])
        truths.append(truth)
    return truths


def get_char_probs(texts, predictions, tokenizer):
    results = [np.zeros(len(t)) for t in texts]
    for i, (text, prediction) in enumerate(zip(texts, predictions)):
        encoded = tokenizer(text, 
                            add_special_tokens=True,
                            return_offsets_mapping=True)
        
        prev_offset = None
        for idx, (offset_mapping, pred) in enumerate(zip(encoded['offset_mapping'], prediction)):
            shift = 0 
            if prev_offset is not None:
                if prev_offset[1] < offset_mapping[0]:
                    shift = 1
            start = offset_mapping[0] - shift
            end = offset_mapping[1]
            results[i][start:end] = pred
            prev_offset = offset_mapping
    return results


def get_results(char_probs, texts, th=0.5):
    results = []
    for char_prob, text in zip(char_probs, texts):
        result = np.where(char_prob >= th)[0] + 1 #Location index for the character that is > threshold / positive prediction
        result = [list(g) for _, g in itertools.groupby(result, key=lambda n, c=itertools.count(): n - next(c))]
        result = [f"{min(r)} {max(r)}" for r in result]

        adjusted_results = []
        for indexes in result:
            start_index, end_index = indexes.split() 
            prev_index = int(start_index) - 1

            if prev_index == 0 and char_prob[prev_index] >= th:
                new_result = str(prev_index) + ' ' + end_index
                adjusted_results.extend([new_result])
            elif char_prob[prev_index] >= th and (text[prev_index] != " " and text[prev_index] != ","):
                new_result = str(prev_index) + ' ' + end_index
                adjusted_results.extend([new_result])
            else:
                adjusted_results.extend([start_index + ' ' + end_index])
        adjusted_results = ";".join(adjusted_results)

        results.append(adjusted_results)
    return results


def get_predictions(results):
    """
    Format the results into a list of [start_index, end_index] for all the positive predictions
    """
    predictions = []
    for result in results:
        prediction = []
        if result != "":
            for loc in [s.split() for s in result.split(';')]:
                start, end = int(loc[0]), int(loc[1])
                prediction.append([start, end])
        predictions.append(prediction)
    return predictions


def convert_token_to_word_probs(texts, predictions, tokenizer, agg='mean'):
    for i, text in enumerate(texts):
        encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
        word_ids = encoded.word_ids()

        counter_start = 0 
        while counter_start < len(word_ids)-1: #-1 here cuz we dont need to care about the [SEP] at the end
            for counter_end in range(counter_start+1, len(word_ids)):
                if word_ids[counter_end] != word_ids[counter_start]:
                    break
                if agg == 'mean':
                    prob = np.mean(predictions[i, counter_start:counter_end])
                elif agg == 'max':
                    prob = np.max(predictions[i, counter_start:counter_end])
                predictions[i, counter_start:counter_end] = prob
            counter_start = counter_end
    return predictions


def get_score(y_true, y_pred):
    score = span_micro_f1(y_true, y_pred)
    return score


def get_logger(filename='inference'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
