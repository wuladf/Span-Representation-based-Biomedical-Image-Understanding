import numpy as np
import json
import logging
from transformers import BertTokenizer

logger = logging.getLogger('root')

def batchify(samples, batch_size):
    """
    Batchfy samples with a batch size
    """
    num_samples = len(samples)

    list_samples_batches = []
    
    # if a sentence is too long, make itself a batch to avoid GPU OOM
    to_single_batch = []
    for i in range(0, len(samples)):
        if len(samples[i]['tokens']) > 350:
            to_single_batch.append(i)
    
    for i in to_single_batch:
        list_samples_batches.append([samples[i]])
    samples = [sample for i, sample in enumerate(samples) if i not in to_single_batch]

    for i in range(0, len(samples), batch_size):
        list_samples_batches.append(samples[i:i+batch_size])

    assert(sum([len(batch) for batch in list_samples_batches]) == num_samples)

    return list_samples_batches

def convert_dataset_to_samples_sent_level(dataset_file, use_gold=True, ner_label2id=None, role_label2id=None):
    f_r = open(dataset_file, 'r', encoding='utf8')
    tokenizer = BertTokenizer.from_pretrained('/your/model/path/')
    samples = []
    if use_gold:    # train or evaluate, the dataset file is train.json or dev.json file
        for line in f_r:
            panel = json.loads(line)
            assert(len(panel['sentences']) == len(panel['spans']))
            for i, sent in enumerate(panel['sentences']):
                sample = {}
                # those two fileds are for recover from sents to panels, 
                # even we shuffle the dataset during training and predicting, 
                # as long as the panels' sents are not split into different dataset part
                sample['panel_id'] = panel['panel_id']
                sample['panel_length'] = panel['panel_length']
                sample['sentence_ix'] = i
                
                # this field is unnessary, since we are already in sentence.
                # however, we can keep some redundant info.
                sample['sent_length'] = len(sent)

                sample['tokens'] = sent
                sample['spans_info'] = panel['spans_info'][i]
                sample['spans'] = []
                sample['spans_ner_label'] = []
                sample['spans_role_label'] = []
                
                for span, span_ner_label, span_role_label in zip(panel['spans'][i], panel['spans_ner_label'][i], panel['spans_role_label'][i]):
                    sample['spans'].append((span[0], span[1], span[1]-span[0]+1))
                    sample['spans_ner_label'].append(ner_label2id[span_ner_label])
                    sample['spans_role_label'].append(role_label2id[span_role_label])
                
                # if the sentence contains no ner
                if not sample['spans']:
                    continue
                
                # use gold means pred_spans and pred_spans_ner_label are the same with...
                sample['pred_spans'] = sample['spans']
                sample['pred_spans_ner_label'] = sample['spans_ner_label']
                
                # filter samples those exceed the model capacity
                if sample['tokens']:
                    tokenized = tokenizer(sample['tokens'], is_split_into_words=True)
                    if len(tokenized.input_ids) < 510:
                        samples.append(sample)
                                                
    else:    # evaluate or predict, the dataset file is dev_prediction.json or test_prediction.json file
        for line in f_r:
            sample = json.loads(line)
            spans = []
            spans_ner_label = []
            spans_role_label = []
            pred_spans = []
            pred_spans_ner_label = []
            
            for span, span_ner_label, span_role_label in zip(sample['spans'], sample['spans_ner_label'], sample['spans_role_label']):
                spans.append((span[0], span[1], span[1]-span[0]+1))
                spans_ner_label.append(ner_label2id[span_ner_label])
                spans_role_label.append(role_label2id[span_role_label])
                
            for pred_span, pred_span_ner_label in zip(sample['pred_spans'], sample['pred_spans_ner_label']):
                pred_spans.append((pred_span[0], pred_span[1], pred_span[1]-pred_span[0]+1))
                pred_spans_ner_label.append(ner_label2id[pred_span_ner_label])
                
                
            sample['spans'] = spans
            sample['spans_ner_label'] = spans_ner_label
            sample['spans_role_label'] = spans_role_label
            sample['pred_spans'] = pred_spans
            sample['pred_spans_ner_label'] = pred_spans_ner_label
            
            if not sample['pred_spans']:
                continue
            
            if sample['tokens']:
                tokenized = tokenizer(sample['tokens'], is_split_into_words=True)
                if len(tokenized.input_ids) < 510:
                    samples.append(sample)
    
    avg_length = sum([len(sample['tokens']) for sample in samples]) / len(samples)
    max_length = max([len(sample['tokens']) for sample in samples])
    logger.info('Extracted %d samples, %.3f avg input length, %d max length'%(len(samples), avg_length, max_length))
            
    return samples
