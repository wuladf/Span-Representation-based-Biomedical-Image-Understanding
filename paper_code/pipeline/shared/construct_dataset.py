from bs4 import BeautifulSoup
import json
from bs4 import element
import nltk
import os

# 1. remove all tag name except <sd-tag>s'
# 2. replace '.' with '$$' for sent split
def preprocess(contents):
    ret = []
    for content in contents:
        if isinstance(content, element.Tag):
            if content.name == 'sd-tag':
                content.string = content.get_text().replace('.', '$$')
                ret.append(content)
            elif content.find('sd-tag'):
                ret.extend(preprocess(content.contents))
            else:
                ret.append(content.get_text())
        else:
            ret.append(content.string)
    return ret

# convert json files to dataset, than split the dataset to train/dev/test set, and save into file
def json2dataset(json_path, output_dir):
    panels = []
    
    for m, panel_file in enumerate(os.listdir(json_path)):
        panel_file_path = os.path.join(json_path, panel_file)
        json_content = open(panel_file_path, mode='r', encoding='utf8')
        json_content = json.load(json_content)
        panel_id = json_content['current_panel_id']
        paper = json_content['paper']
        doi = paper['doi']
        figure = json_content['figure']
        ps = figure['panels']
        label = figure['label']
        # ensure the panel id is unique
        panel_id = str(doi) + '_' + str(panel_id) + '_' + str(label) + '_' + str(m+1)
        
        for n, pa in enumerate(ps):
            panel = {}
            panel['panel_id'] = panel_id + '_' + str(n+1)
            panel['sentences'] = []
            panel['spans'] = []
            
            caption = pa['formatted_caption']
            # for better sent split
            caption = caption.replace('s.d.', 'sd').replace('s.e.m.','sem')
            soup = BeautifulSoup(caption, 'lxml')
            
            contents = preprocess(soup.contents)
            text = ''.join(str(content) for content in contents)
            
            for sent in nltk.sent_tokenize(text):
                sent = sent.replace('$$', '.')
                spans = []
                tokens = []
                sent = BeautifulSoup(sent, 'lxml')
                tag = sent.find('body')
                if sent.find('p'):
                    tag = sent.find('p')
                if not tag:
                    continue
                
                for content in tag.contents:
                    tks = nltk.word_tokenize(content.get_text())
                    if isinstance(content, element.Tag):
                        if content.get('type'):
                            if content['type'] != 'undefined':
                                span = []
                                etype = content['type']
                                role = content['role']
                                mid = content.get('external_id0')
                                span.append(len(tokens))
                                tokens.extend(tks)
                                span.append(len(tokens)-1)
                                span.append(etype)
                                span.append(role)
                                span.append(str(mid))
                                span.append(content.get_text())
                                spans.append(span)
                            else:
                                tokens.extend(tks)
                        else:
                            tokens.extend(tks)
                    else:
                        tokens.extend(tks)
                
                panel['sentences'].append(tokens)
                panel['spans'].append(spans)
                
            panels.append(panel)
            
#     train_panels = panels[0 : int(len(panels)*0.6)]
    output_file = os.path.join(output_dir, 'train.json')
    f_w = open(output_file, 'w')
    f_w.write('\n'.join(json.dumps(panel) for panel in panels[0 : int(len(panels)*0.6)]))
    
#     dev_panels = panels[int(len(panels)*0.6) : int(len(panels)*0.8)]
    output_file = os.path.join(output_dir, 'dev.json')
    f_w = open(output_file, 'w')
    f_w.write('\n'.join(json.dumps(panel) for panel in panels[int(len(panels)*0.6) : int(len(panels)*0.8)]))
    
#     test_panels = panels[int(len(panels)*0.8) : ]
    output_file = os.path.join(output_dir, 'test.json')
    f_w = open(output_file, 'w')
    f_w.write('\n'.join(json.dumps(panel) for panel in panels[int(len(panels)*0.8) : ]))
