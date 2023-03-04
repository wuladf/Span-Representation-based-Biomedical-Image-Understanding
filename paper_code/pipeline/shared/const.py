ner_labels = ['nan', 'gene', 'protein', 'molecule', 'subcellular', 'cell', 'tissue', 'organism']

role_labels =  ['component', 'assayed', 'intervention', 'reporter', 'experiment', 'normalizing']

def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i
        id2label[i] = label
    return label2id, id2label