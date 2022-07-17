import random
from dataclasses import dataclass
import math
import datasets
from typing import Union, List, Tuple, Dict
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import Dataset

# from .arguments import DataArguments, RerankerTrainingArguments
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers import DataCollatorWithPadding
import numpy as np
from tqdm import tqdm
import os

import random
from dataclasses import dataclass
import math
import datasets
from typing import Union, List, Tuple, Dict
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import Dataset

# from .arguments import DataArguments, RerankerTrainingArguments
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers import DataCollatorWithPadding
import numpy as np
from tqdm import tqdm
import os

class HBERTPretrainedPointWiseDataset(Dataset):
    def __init__(
            self,
            args,
            tokenizer: PreTrainedTokenizer,
            dataset_cache_dir,
            dataset_script_dir,
            max_seq_len,
    ):
        self.max_seq_len = max_seq_len
        train_file =os.path.abspath(args.train_file) 
        if os.path.isdir(train_file):
            filenames = os.listdir(train_file)
            train_files = [os.path.join(train_file, fn) for fn in filenames]
        else:
            train_files = train_file
        block_size_10MB = 10<<20
        print("start loading datasets, train_files: ", train_files)
        # print(dataset_script_dir)
        self.nlp_dataset = datasets.load_dataset(
            f'{dataset_script_dir}/json.py',
            data_files=train_files,
            ignore_verifications=False,
            cache_dir=dataset_cache_dir,
            features=datasets.Features({
                "text_tokens_idx":[datasets.Value("int32")],
                "node_tokens_idx":[datasets.Value("int32")],
                "inputs_type_idx":[datasets.Value("int32")],
                "text_labels":[datasets.Value("int32")],
                "node_labels":[datasets.Value("int32")],
                "text_layer_index":[datasets.Value("int32")],
                "node_layer_index":[datasets.Value("int32")],
                "text_num":[datasets.Value("int32")],
                "node_num":[datasets.Value("int32")],
                "waiting_mask":[datasets.Value("int32")],
                "position":[datasets.Value("int32")],
            }),
            block_size = block_size_10MB
        )['train']

        
        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.args = args
        self.total_len = len(self.nlp_dataset)

        print("loading dataset ok! len of dataset,", self.total_len)
    
    def __len__(self):
        return self.total_len

    
    def __getitem__(self, item):
        data = self.nlp_dataset[item]
        max_seq_len = self.max_seq_len

        text_num = data['text_num']
        node_num = data['node_num']
        position_len = max([text_num[i]+node_num[i] for i in range(len(text_num))])

        data = {
                "token_input_ids":torch.LongTensor(data['text_tokens_idx']),
                "node_input_ids":torch.LongTensor(data['node_tokens_idx']),
                "inputs_type_idx":torch.LongTensor(data['inputs_type_idx']),
                "token_labels":torch.LongTensor(data['text_labels']),
                "node_labels":torch.LongTensor(data['node_labels']),
                "token_layer_index":torch.LongTensor(data['text_layer_index']),
                "node_layer_index":torch.LongTensor(data['node_layer_index']),
                "seq_num":list(data['text_num']),
                "node_num":list(data['node_num']),
                "waiting_mask":torch.LongTensor(data['waiting_mask']),
                "position":list(data['position']),
                "position_len":position_len,
        }
        return BatchEncoding(data)

@dataclass
class HBERTPointCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    def __call__(
            self, features
    ):
        
        max_seq_len = 256
        max_node_len = 10
        # print(features)
        batch_size = len(features)
        mlm_labels = []
        inputs_type_idx = []
        layer_index = []
        waiting_mask = []
        position = []
        layer_num = len(features[0]['seq_num'])
        token_input_ids = []
        node_input_ids = []
        inputs_type_idx = []
        token_labels= []
        node_labels= []
        token_layer_index= []
        node_layer_index= []
        seq_num= []
        node_num= []
        waiting_mask= []
        for i in range(batch_size):
            one_position = features[i]['position']
            position_len = features[i]['position_len']
            one_position = [torch.LongTensor(one_position[j:j+position_len]) for j in range(0, len(one_position),position_len )]
            assert len(one_position) == layer_num
            position.extend(one_position)
            token_input_ids.append(features[i]['token_input_ids'])
            node_input_ids.append(features[i]['node_input_ids'])
            inputs_type_idx.append(features[i]['inputs_type_idx'])
            token_labels.append(features[i]['token_labels'])
            token_layer_index.append(features[i]['token_layer_index'])
            node_layer_index.append(features[i]['node_layer_index'])
            seq_num.append(features[i]['seq_num'])
            node_num.append(features[i]['node_num'])
            waiting_mask.append(features[i]['waiting_mask'])
            node_labels.append(features[i]['node_labels'])

            del features[i]['token_input_ids']
            del features[i]['node_input_ids']
            del features[i]['inputs_type_idx']
            del features[i]['token_labels']
            del features[i]['node_labels']
            del features[i]['token_layer_index']
            del features[i]['node_layer_index']
            del features[i]['seq_num']
            del features[i]['node_num']
            del features[i]['waiting_mask']
            del features[i]['position']
            del features[i]['position_len']
        features = {}
        features["token_input_ids"] = pad_sequence(token_input_ids, batch_first=True).view(batch_size,-1,max_seq_len)
        features["node_input_ids"] = pad_sequence(node_input_ids, batch_first=True).view(batch_size,-1,max_node_len)
        features["inputs_type_idx"] = pad_sequence(inputs_type_idx, batch_first=True).view(batch_size,-1,max_seq_len)
        features["token_labels"] = pad_sequence(token_labels, batch_first=True).view(batch_size,-1,max_seq_len)
        features["node_labels"] = pad_sequence(node_labels, batch_first=True).view(batch_size,-1,max_node_len)
        features["token_layer_index"] = pad_sequence(token_layer_index, batch_first=True)
        features["node_layer_index"] = pad_sequence(node_layer_index, batch_first=True)
        features["seq_num"] = torch.LongTensor(seq_num)
        features["node_num"] = torch.LongTensor(node_num)
        features["waiting_mask"] = pad_sequence(waiting_mask, batch_first=True).view(batch_size,-1,max_node_len)
        features["position"] = pad_sequence(position, batch_first=True).view(batch_size,layer_num,-1)
        return features




    
