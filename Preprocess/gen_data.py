import os
from argparse import ArgumentParser
import sys
sys.path.append('../')
from tqdm import tqdm
import json
from transformers import BertTokenizer
import numpy as np
from random import random, shuffle, choice, sample
import collections
import traceback
from multiprocessing import Pool, Value, Lock
from tempfile import TemporaryDirectory
from pathlib import Path
import shelve
from collections import Counter
import math
import copy
import sys
sys.setrecursionlimit(10000)
MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])
lock = Lock()
num_instances = Value('i', 0)
num_docs = Value('i', 0)
num_words = Value('i', 0)

TEMP_DIR = '../'

from transformers import BertTokenizer, BertModel
BertPath = '../bert_base_uncased'
tag_vocab_path = '/home/yu_guo/DataPreProcess/data/tag.vocab'
bert_tokenizer = BertTokenizer.from_pretrained(BertPath)

def get_text(html,i):
    depth = 0
    word = []
    type_idx = []
    tag_name = html[i]["name"]
    tag_children = html[i]["children"]
    tag_text = html[i]["text"]
    tag_idx = html[i]["id"]
    if tag_name == "textnode":
        res = bert_tokenizer.tokenize(tag_text)
        return res,[0]*len(res),depth
    else:
        for child_idx in tag_children:
            inner_word,inner_type_idx,tag_depth = get_text(html,int(child_idx))
            word += inner_word
            type_idx += inner_type_idx
            depth = max(depth,tag_depth)
        depth += 1
        assert len(type_idx)==len(word)
        return ["<"+tag_name+">"]+word+["<"+tag_name+">"],[1]+type_idx+[2],depth

def get_position(x):
    res = {}
    out = []
    for i in range(len(x)):
        res[x[i]] = i
    for i in range(len(x)):
        out.append(res[i])
    return out

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
	"""Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_num_tokens:
			break
		# truncate from the doc side
		tokens_b.pop()
def construct_pointwise_data(examples,chunk_indexs,max_seq_len,bert_tokenizer,masked_lm_prob,
            max_predictions_per_seq,bert_vocab_list,epoch_filename,resps_list):
	with open(epoch_filename,'w')as g:
		num_examples = len(examples)
		print("num_examples", num_examples)
		num_instance = 0
		num_instances_value = 0
		for doc_idx in tqdm(chunk_indexs):
		# print(doc_idx)
			if doc_idx % 100 == 0:
				print(doc_idx)
			example = examples[doc_idx]

			instances = [] 
			post_tokens = example['post']
			resp_tokens = example['response']
			neg_resp_tokens = select_neg_resps(resps_list,post_tokens,resp_tokens)

			pos_post = copy.deepcopy(post_tokens)[:250]
			pos_resp = copy.deepcopy(resp_tokens)[:250]
			neg_post = copy.deepcopy(post_tokens)[:250]
			neg_resp = copy.deepcopy(neg_resp_tokens)[:250]
			try:
				truncate_seq_pair(pos_post, pos_resp, max_seq_len - 3)
				truncate_seq_pair(neg_post, neg_resp, max_seq_len - 3)
			except:
				break

			pos_tokens = ["[CLS]"] + pos_post + ["[SEP]"] + pos_resp + ["[SEP]"]
			pos_segment_ids = [0 for _ in range(len(pos_post) + 2)] + [1 for _ in range(len(pos_resp) + 1)]
			neg_tokens = ["[CLS]"] + neg_post + ["[SEP]"] + neg_resp + ["[SEP]"]
			neg_segment_ids = [0 for _ in range(len(neg_post) + 2)] + [1 for _ in range(len(neg_resp) + 1)]
			pos_tokens, pos_masked_lm_positions, pos_masked_lm_labels = create_masked_lm_predictions(
				pos_tokens, masked_lm_prob, max_predictions_per_seq, True, bert_vocab_list)
			neg_tokens, neg_masked_lm_positions, neg_masked_lm_labels = create_masked_lm_predictions(
				neg_tokens, masked_lm_prob, max_predictions_per_seq, True, bert_vocab_list)
			pos_tokens_idx = bert_tokenizer.convert_tokens_to_ids(pos_tokens)
			pos_tokens_idx_labels = bert_tokenizer.convert_tokens_to_ids(pos_masked_lm_labels)
			neg_tokens_idx = bert_tokenizer.convert_tokens_to_ids(neg_tokens)
			neg_tokens_idx_labels = bert_tokenizer.convert_tokens_to_ids(neg_masked_lm_labels)
			pos_instance = {
				"tokens_idx":pos_tokens_idx,
				"tokens":pos_tokens,
				"segment_ids": pos_segment_ids,
				"label": 1,
				"masked_lm_positions": pos_masked_lm_positions,
				"masked_lm_labels_idxs": pos_tokens_idx_labels,
				}
			neg_instance = {
				"tokens_idx":neg_tokens_idx,
				"tokens":neg_tokens,
				"segment_ids": neg_segment_ids,
				"label": 0,
				"masked_lm_positions": neg_masked_lm_positions,
				"masked_lm_labels_idxs": neg_tokens_idx_labels,
				}
			g.write(json.dumps(pos_instance, ensure_ascii=False)+'\n')
			g.write(json.dumps(neg_instance, ensure_ascii=False)+'\n')
			num_instances_value += 1

	return num_instances_value
def pad_positions(positions,max_num):
    for i in range(len(positions)):
        positions[i] = positions[i]+[0]*(max_num-len(positions[i]))
    return positions
def get_layer_info(nodes,max_layer_num,node_max_length,seq_max_length):
    tag = 1
    node_queue=[]
    nodes_info = [] 
    texts_info = []
    attention_mask = []
    nodes[0]["depth"] = 1
    node_queue.append(nodes[0])
    node_layer_index = []
    text_layer_index = []
    waiting_masks = []
    positions = []
    node_position = []
    text_position = []
    last_depth = 1
    text_num = []
    node_num = []
    current_idx = 0 
    max_layer_length = 0 
    while(len(node_queue)>0):
        if node_queue[0]['depth'] != last_depth and node_queue[0]['depth'] <= max_layer_num:
            node_num.append(len(node_position))
            text_num.append(len(text_position))
            node_position.extend(text_position)
            node_position = get_position(node_position)
            max_layer_length = max(max_layer_length,len(node_position))
            positions.append(node_position)
            node_position = []
            text_position = []
            current_idx = 0
            last_depth = node_queue[0]['depth']
        tag_idx = node_queue[0]['id']
        if node_queue[0]["name"] == "textnode" or node_queue[0]["depth"] == max_layer_num :
            text_info = {}
            text,type_idx,_ = get_text(nodes,tag_idx)
            text_layer_index.append(node_queue[0]["depth"])
            node_queue = node_queue[1:]
            text_info['text'] = text[:seq_max_length-1]
            text_info['type_idx'] = type_idx[:seq_max_length-1]
            text_position.append(current_idx)
            current_idx+=1
            texts_info.append(text_info)
        else:
            node_info = {}
            tag_name = node_queue[0]["name"]
            children = node_queue[0]["children"]
            meaning_children  = []
            for child in children:
                if nodes[child]["name"] == 'textnode' and nodes[child]['text'].strip()=="":
                    continue
                else:
                    meaning_children.append(child)
            children = meaning_children
            children = children[:node_max_length-2]
            children_num = len(children)
            if children_num == 0:
                tag = 0
                break
            for child in children:
                nodes[child]["depth"]=node_queue[0]["depth"]+1
                node_queue.append(nodes[child])
            text=['[CLS]','<'+tag_name+'>']
            #type_idx=[0,1]
            node_layer_index.append(node_queue[0]["depth"])
            node_position.append(current_idx)
            current_idx+=1
            waiting_mask = [0]*2 + [1]*children_num + [0] * max(node_max_length-(children_num+2),0)
            waiting_mask = waiting_mask[:node_max_length]
            node_queue = node_queue[1:]
            node_info['text'] = text
            nodes_info.append(node_info)
            waiting_masks.append(waiting_mask)
    if tag==0:
        return None
    text_num.append(len(text_position))
    node_num.append(len(node_position))
    node_position.extend(text_position)
    node_position = get_position(node_position)
    positions.append(node_position)
    max_layer_length = max(max_layer_length,len(node_position))
    if len(text_num) < max_layer_num:
        for i in range(max_layer_num - len(node_num)):
            positions.append([])
        text_num += [0] * (max_layer_num-len(text_num))
        node_num += [0] * (max_layer_num - len(node_num))
    positions = pad_positions(positions,max_layer_length)
    assert len(positions) == max_layer_num
    assert len(text_num) == max_layer_num
    assert len(node_num) == max_layer_num
    # nodes_info = nodes_info[:node_max_length]
    # layer_index = layer_index[:node_max_length]
    # waiting_masks = waiting_masks[:node_max_length]
    res = {
        "nodes_info":nodes_info,
        "texts_info":texts_info,
        "node_layer_index":node_layer_index,
        "text_layer_index":text_layer_index,
        "waiting_mask":waiting_masks,
        "node_num":node_num,
        "text_num":text_num,
        "position":positions
    }
    return res
            
        
def load_vocab(path):
    vocab = []
    with open(path,'r')as f:
        for one_tag in f:
            vocab.append(one_tag.strip())
    return vocab
# model

class DocumentDatabase:
    def __init__(self, reduce_memory=False):
        if reduce_memory:
            self.temp_dir = TemporaryDirectory(dir=TEMP_DIR)
            self.working_dir = Path(self.temp_dir.name)
            self.document_shelf_filepath = self.working_dir / 'shelf.db'
            self.document_shelf = shelve.open(str(self.document_shelf_filepath),
                                              flag='n', protocol=-1)
            self.documents = None
        else:
            self.documents = []
            self.document_shelf = None
            self.document_shelf_filepath = None
            self.temp_dir = None
        self.doc_lengths = []
        self.doc_cumsum = None
        self.cumsum_max = None
        self.reduce_memory = reduce_memory

    def add_document(self, document):
        if not document:
            return
        if self.reduce_memory:
            current_idx = len(self.doc_lengths)
            self.document_shelf[str(current_idx)] = document
        else:
            self.documents.append(document)
        self.doc_lengths.append(len(document))

    def __len__(self):
        return len(self.doc_lengths)

    def __getitem__(self, item):
        if self.reduce_memory:
            return self.document_shelf[str(item)]
        else:
            return self.documents[item]

    def __contains__(self, item):
        if str(item) in self.document_shelf:
            return True
        else:
            return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()

def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    # [MASK] word from DOC, not the query
    for (i, token) in enumerate(tokens):
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (whole_word_mask and len(cand_indices) >= 1 and token.startswith("##")):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])
    num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(cand_indices) * masked_lm_prob))))
    shuffle(cand_indices)
    mask_indices = sorted(sample(cand_indices, num_to_mask))
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_mask:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = choice(vocab_list)
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
            tokens[index] = masked_token

    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]
    return tokens, mask_indices, masked_token_labels

def construct_pointwise_examples(examples,chunk_indexs,max_seq_len,max_tag_len,max_layer_num,mlm,bert_tokenizer,masked_lm_prob,
                max_predictions_per_seq,bert_vocab_list,epoch_filename,word2df,mu,total_doc_cnt):
    num_examples = len(examples)
    num_instance = 0
    wrong = 0
    for doc_idx in tqdm(chunk_indexs):
        example = examples[doc_idx]
        layer_info = get_layer_info(example,max_layer_num=max_layer_num,node_max_length=max_tag_len,seq_max_length = max_seq_len)
        if layer_info is None:
            wrong+=1
            continue
        nodes_info = layer_info["nodes_info"]
        texts_info = layer_info["texts_info"]
        node_layer_index = layer_info["node_layer_index"]
        text_layer_index = layer_info["text_layer_index"]
        waiting_mask = layer_info["waiting_mask"]
        node_num = layer_info["node_num"]
        text_num = layer_info["text_num"]
        position = layer_info["position"]

        tag_len = len(node_layer_index)
        
        res={}
        instances = []
        nodes_tokens = []
        nodes_tokens_idx = []
        texts_tokens = []
        texts_tokens_idx = []
        texts_type_idxs = []
        text_labels = []
        node_labels = []

        for node_info in nodes_info:
            tokens = node_info['text']
            if mlm:
                tokens,masked_lm_positions,masked_lm_labels = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, True, bert_vocab_list)
            else:
                masked_lm_positions, masked_lm_labels = [], []
            tokens_idx = bert_tokenizer.convert_tokens_to_ids(tokens)
            tokens_idx_labels = bert_tokenizer.convert_tokens_to_ids(masked_lm_labels)
            nodes_tokens.append(tokens+(max_tag_len-len(tokens))*['PAD'])
            nodes_tokens_idx.append(tokens_idx+(max_tag_len-len(tokens))*[0])
            node_label = np.array([-100] * max_tag_len)
            node_label[masked_lm_positions] = tokens_idx_labels
            node_labels.append(node_label.tolist())
        for node_info in texts_info:
            text = node_info['text']
            type_idxs = node_info['type_idx']
            tokens = ["[CLS]"] + text
            tokens = tokens[:max_seq_len]
            type_idxs = [0] + type_idxs
            type_idxs = type_idxs[:max_seq_len]
            if mlm:
                tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, True, bert_vocab_list)
            else:
                masked_lm_positions, masked_lm_labels = [], []
            # else:
            #     tokens = node_info['text']
            #     type_idxs = node_info['type_idx']
            #     masked_lm_positions, masked_lm_labels = [], []
            tokens_idx = bert_tokenizer.convert_tokens_to_ids(tokens)
            tokens_idx_labels = bert_tokenizer.convert_tokens_to_ids(masked_lm_labels)
            texts_tokens.append(tokens+(max_seq_len-len(tokens))*['PAD'])
            texts_tokens_idx.append(tokens_idx+(max_seq_len-len(tokens))*[0])
            texts_type_idxs.append(type_idxs+(max_seq_len-len(tokens))*[0])
            text_label = np.array([-100] * max_seq_len)
            text_label[masked_lm_positions] = tokens_idx_labels
            text_labels.append(text_label.tolist())
        instance = {
            "node_tokens_idx":sum(nodes_tokens_idx,[]),
            "text_tokens_idx":sum(texts_tokens_idx,[]),
            "inputs_type_idx":sum(texts_type_idxs,[]),
            "node_labels":sum(node_labels,[]),
            "text_labels":sum(text_labels,[]),
            "node_layer_index":list(node_layer_index),
            "text_layer_index":list(text_layer_index),
            "waiting_mask":sum(waiting_mask,[]),
            "position":sum(position,[]),
            "node_num":list(node_num),
            "text_num":list(text_num),
        }
        doc_instances=json.dumps(instance,ensure_ascii=False)
        lock.acquire()
        with open(epoch_filename,'a+') as epoch_file:
            epoch_file.write(doc_instances + '\n')
            num_instances.value += 1
        lock.release()
    print("wrong:",wrong)
def error_callback(e):
    print('error')
    print(dir(e), "\n")
    traceback.print_exception(type(e), e, e.__traceback__)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=str, required=True)
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--bert_model", type=str, default='bert-base-uncased')
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")
    parser.add_argument("--epochs_to_generate", type=int, default=1,
                        help="Number of epochs of data to pregenerate")
    # parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--max_tag_len", type=int, default=10)
    parser.add_argument("--max_layer_num",type=int, default=5)
    parser.add_argument("--mlm", action="store_true")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=60,
                        help="Maximum number of tokens to mask in each sequence")
    parser.add_argument("--rop_num_per_doc", type=int, default=1,
                        help="How many samples for each document")
    parser.add_argument("--pairnum_per_doc", type=int, default=2,
                        help="How many samples for each document")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="The number of workers to use to write the files")
    parser.add_argument("--mu", type=int, default=512,
                        help="The number of workers to use to write the files")
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()

    bert_tokenizer = BertTokenizer.from_pretrained(BertPath)
    bert_model = BertModel.from_pretrained(BertPath)

    # ADDITIONAL_SPECIAL_TOKENS = load_vocab(tag_vocab_path)
    # bert_tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})

    #bert_tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    bert_model.resize_token_embeddings(len(bert_tokenizer))
    bert_param_ids = list(map(id, bert_model.parameters()))
    bert_vocab_list = list(bert_tokenizer.vocab.keys())

    # word2df = {}
    # with open(os.path.join(args.output_dir, f"word2df.json"), 'r') as wdf:
    # word2df = json.loads(wdf.read().strip())
    word2df = None
    examples = []
    with DocumentDatabase(reduce_memory=args.reduce_memory) as docs:
        with open(args.train_corpus,encoding='UTF-8') as f:
            idx = 0
            for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
                idx+=1
                example = json.loads(line.strip())
                #example = json.loads(line)
                # examples.append(example)
                docs.add_document(example)

        print('Reading file is done! Total doc num:{}'.format(len(docs)))




        for epoch in range(args.epochs_to_generate):
            epoch_filename =  f"{args.output_dir}/epoch_{epoch}.json"
            if os.path.exists(epoch_filename):
                with open(epoch_filename, "w") as ef:
                    print(f"start generating {epoch_filename}")
            num_processors = args.num_workers
            processors = Pool(num_processors)
            cand_idxs = list(range(0, len(docs)))

            for i in range(num_processors):
                chunk_size = int(len(cand_idxs) / num_processors)
                chunk_indexs = cand_idxs[i*chunk_size:(i+1)*chunk_size]
                # print("?")
                r = processors.apply_async(construct_pointwise_examples, (docs, chunk_indexs, args.max_seq_len,args.max_tag_len,args.max_layer_num, args.mlm, bert_tokenizer, args.masked_lm_prob, \
                args.max_predictions_per_seq, bert_vocab_list, epoch_filename, word2df, args.mu, len(docs)), error_callback=error_callback)
            processors.close()
            processors.join()



            metrics_file =  f"{args.output_dir}/epoch_{epoch}_metrics.json"
            with open(metrics_file, 'w') as metrics_file:
                metrics = {
                    "num_training_examples": num_instances.value,
                    "max_seq_len": args.max_seq_len
                }
                metrics_file.write(json.dumps(metrics))
