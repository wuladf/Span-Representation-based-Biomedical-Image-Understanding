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

def convert_dataset_to_samples_sent_level(dataset_file, max_span_length, ner_label2id=None, role_label2id=None):
    f_w = open(dataset_file, 'r', encoding='utf8')
    tokenizer = BertTokenizer.from_pretrained('/root/autodl-nas/PubMedBERT/')
    samples = []
    for line in f_w:
        panel = json.loads(line)
        assert(len(panel['sentences']) == len(panel['spans']))
        for i, sent in enumerate(panel['sentences']):
            sample = {}
            # those fileds are for recover from sents to panels, 
            # even we shuffle the dataset during training and predicting, 
            # as long as the panels' sents are not split into different dataset part
            sample['panel_id'] = panel['panel_id']
            sample['panel_length'] = panel['panel_length']
            sample['sentence_ix'] = i
            
            # this field is unnessary, since we are already in sentence.
            # however, we can keep some redundant info.
            sample['sent_length'] = len(sent)
            
            sample['tokens'] = sent
            if sample['tokens']:
                tokenized = tokenizer(sample['tokens'], is_split_into_words=True)
                if len(tokenized.input_ids) > 510:
                    continue
            if not sample['tokens']:
                continue
                
            sample['spans_info'] = panel['spans_info'][i]
            sample['spans'] = []
            sample['spans_ner_label'] = []
#             sample['spans_role_label'] = panel['spans_role_label'][i]
            sample['spans_role_label'] = []
            
            spans = panel['spans'][i]
            ners = panel['spans_ner_label'][i]
            roles = panel['spans_role_label'][i]
#             for span in panel['spans'][i]:
#                 spans.append((span[0], span[1]))
#                 ners.append(span[2])
#                 roles.append(span[3])
            
            for j in range(len(sent)):
                for k in range(j, min(len(sent), j+max_span_length)):
                    sample['spans'].append((j, k, k-j+1))
                    if [j, k] not in spans:
                        sample['spans_ner_label'].append(0)
                        sample['spans_role_label'].append(0)
                    else:
                        idx = spans.index([j, k])
                        sample['spans_ner_label'].append(ner_label2id[ners[idx]])
                        sample['spans_role_label'].append(role_label2id[roles[idx]])
            samples.append(sample)
    
    avg_length = sum([len(sample['tokens']) for sample in samples]) / len(samples)
    max_length = max([len(sample['tokens']) for sample in samples])
    logger.info('Extracted %d samples, %.3f avg input length, %d max length'%(len(samples), avg_length, max_length))
            
    return samples