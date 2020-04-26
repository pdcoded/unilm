"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
import logging
import argparse
import math
from tqdm import tqdm, trange
import numpy as np
import torch
import random
import pickle

from transformers import BertTokenizer, RobertaTokenizer
from services.modeling_decoding import BertForSeq2SeqDecoder, BertConfig
from transformers.tokenization_bert import whitespace_tokenize
import services.s2s_loader as seq2seq_loader
from services.utils import load_and_cache_examples
from transformers import \
    BertTokenizer, RobertaTokenizer
from services.tokenization_unilm import UnilmTokenizer
from services.tokenization_minilm import MinilmTokenizer

TOKENIZER_CLASSES = {
    'bert': BertTokenizer,
    'minilm': MinilmTokenizer,
    'roberta': RobertaTokenizer,
    'unilm': UnilmTokenizer,
}

class WhitespaceTokenizer(object):
    def tokenize(self, text):
        return whitespace_tokenize(text)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)


def main(request_body):

    if request_body=={"context":"","answer":""}:
      return ("Nothing to extract questions from")

    else:
        input_text = request_body["context"] + " [SEP] " + request_body["answer"]

        #bert_model = "bert-large-cased"
        #model_path = "../output_dir/ckpt-48000"

        #input_text = request_body["context"] + " [SEP] " + request_body["answer"]
        parser = argparse.ArgumentParser()
        #model_type = "minilm"

        #tokenizer_name = "minilm-l12-h384-uncased"


        # Required parameters

        parser.add_argument("--model_type", default="minilm", type=str, required=False,help="Model type selected in the list: " + ", ".join(TOKENIZER_CLASSES.keys()))
        
        parser.add_argument("--model_path", default="./output_dir/", type=str, required=False,help="Path to the model checkpoint.")
        
        parser.add_argument("--config_path", default="./services/minilm_config.json", type=str,help="Path to config.json for the model.")

        # tokenizer_name
        parser.add_argument("--tokenizer_name", default="minilm-l12-h384-uncased", type=str, required=False, 
                            help="tokenizer name")
        parser.add_argument("--max_seq_length", default=64, type=int,
                            help="The maximum total input sequence length after WordPiece tokenization. \n"
                                 "Sequences longer than this will be truncated, and sequences shorter \n"
                                 "than this will be padded.")

        # decoding parameters
        """
        parser.add_argument('--fp16', action='store_false',
                            help="Whether to use 16-bit float precision instead of 32-bit")

        parser.add_argument('--amp', action='store_false',
                            help="Whether to use amp for fp16")

        """
        parser.add_argument("--input_file", type=str, help="Input file")
        parser.add_argument('--subset', type=int, default=0,
                            help="Decode a subset of the input dataset.")
        parser.add_argument("--output_file", type=str, help="output file")
        parser.add_argument("--split", type=str, default="",
                            help="Data split (train/val/test).")
        parser.add_argument('--tokenized_input', action='store_true',
                            help="Whether the input is tokenized.")
        parser.add_argument('--seed', type=int, default=123,
                            help="random seed for initialization")
        parser.add_argument("--do_lower_case", action='store_true',
                            help="Set this flag if you are using an uncased model.")
        parser.add_argument('--batch_size', type=int, default=4,
                            help="Batch size for decoding.")
        parser.add_argument('--beam_size', type=int, default=1,
                            help="Beam size for searching")
        parser.add_argument('--length_penalty', type=float, default=0,
                            help="Length penalty for beam search")

        parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
        parser.add_argument('--forbid_ignore_word', type=str, default=None,
                            help="Forbid the word during forbid_duplicate_ngrams")
        parser.add_argument("--min_len", default=1, type=int)
        parser.add_argument('--need_score_traces', action='store_true')
        parser.add_argument('--ngram_size', type=int, default=3)
        parser.add_argument('--mode', default="s2s",
                            choices=["s2s", "l2r", "both"])
        parser.add_argument('--max_tgt_length', type=int, default=48,
                            help="maximum length of target sequence")
        parser.add_argument('--s2s_special_token', action='store_true',
                            help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
        parser.add_argument('--s2s_add_segment', action='store_true',
                            help="Additional segmental for the encoder of S2S.")
        parser.add_argument('--s2s_share_segment', action='store_true',
                            help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
        parser.add_argument('--pos_shift', action='store_true',
                            help="Using position shift for fine-tuning.")
        parser.add_argument("--cache_dir", default="./cache_dir", type=str,
                            help="Where do you want to store the pre-trained models downloaded from s3")

        args = parser.parse_args()

        if args.need_score_traces and args.beam_size <= 1:
            raise ValueError(
                "Score trace is only available for beam search with beam size > 1.")
        if args.max_tgt_length >= args.max_seq_length - 2:
            raise ValueError("Maximum tgt length exceeds max seq length - 2.")

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()

        if args.seed > 0:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if n_gpu > 0:
                torch.cuda.manual_seed_all(args.seed)
        else:
            random_seed = random.randint(0, 10000)
            logger.info("Set random seed as: {}".format(random_seed))
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            if n_gpu > 0:
                torch.cuda.manual_seed_all(args.seed)
        
        tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
            args.tokenizer_name, do_lower_case=args.do_lower_case, 
            cache_dir=args.cache_dir if args.cache_dir else None)

        if args.model_type == "roberta":
            vocab = tokenizer.encoder
        else:
            vocab = tokenizer.vocab

        tokenizer.max_len = args.max_seq_length

        config_file = args.config_path if args.config_path else os.path.join(args.model_path, "config.json")
        logger.info("Read decoding config from: %s" % config_file)
        config = BertConfig.from_json_file(config_file)

        bi_uni_pipeline = []
        bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(
            list(vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length,
            max_tgt_length=args.max_tgt_length, pos_shift=args.pos_shift,
            source_type_id=config.source_type_id, target_type_id=config.target_type_id, 
            cls_token=tokenizer.cls_token, sep_token=tokenizer.sep_token, pad_token=tokenizer.pad_token))

        print ("bi_uni_pipeline is ",bi_uni_pipeline[0](([100, 2003, 1037, 100, 100, 1999, 100, 102, 100],9)) )

        mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
            [tokenizer.mask_token, tokenizer.sep_token, tokenizer.sep_token])
        forbid_ignore_set = None
        if args.forbid_ignore_word:
            w_list = []
            for w in args.forbid_ignore_word.split('|'):
                if w.startswith('[') and w.endswith(']'):
                    w_list.append(w.upper())
                else:
                    w_list.append(w)
            forbid_ignore_set = set(tokenizer.convert_tokens_to_ids(w_list))
        print(args.model_path)
        found_checkpoint_flag = False
        for model_recover_path in [args.model_path.strip()]:
            logger.info("***** Recover model: %s *****", model_recover_path)
            found_checkpoint_flag = True
            model = BertForSeq2SeqDecoder.from_pretrained(
                model_recover_path, config=config, mask_word_id=mask_word_id, search_beam_size=args.beam_size,
                length_penalty=args.length_penalty, eos_id=eos_word_ids, sos_id=sos_word_id,
                forbid_duplicate_ngrams=args.forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set,
                ngram_size=args.ngram_size, min_len=args.min_len, mode=args.mode,
                max_position_embeddings=args.max_seq_length, pos_shift=args.pos_shift, 
            )
            """
            if args.fp16:
                model.half()
            model.to(device)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)
            """
            torch.cuda.empty_cache()
            model.eval()
            next_i = 0
            max_src_length = args.max_seq_length - 2 - args.max_tgt_length

            source_tokens = tokenizer.tokenize(input_text)
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)[:max_src_length]
            input_lines = source_ids

            output_lines = [""] * len(input_lines)
            instances=[]
            instances.append(bi_uni_pipeline[0]((input_lines,len(input_lines))))
            with torch.no_grad():
                batch = seq2seq_loader.batch_list_to_batch_tensors(
                    instances)
                batch = [
                    t.to(device) if t is not None else None for t in batch]
                input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
                traces = model(input_ids, token_type_ids,
                               position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv)
                if args.beam_size > 1:
                    traces = {k: v.tolist() for k, v in traces.items()}
                    output_ids = traces['pred_seq']
                else:
                    output_ids = traces.tolist()
                
                w_ids = output_ids[0]
                output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                output_tokens = []
                for t in output_buf:
                    if t in (tokenizer.sep_token, tokenizer.pad_token):
                        break
                    output_tokens.append(t)
                if args.model_type == "roberta":
                    output_sequence = tokenizer.convert_tokens_to_string(output_tokens)
                else:
                    output_sequence = ' '.join(detokenize(output_tokens))
                if '\n' in output_sequence:
                    output_sequence = " [X_SEP] ".join(output_sequence.split('\n'))
                output_lines= output_sequence
        
        return (output_lines)
    
    
    


if __name__ == "__main__":
    print (main({"context":"PD is a Solutions Architect in Quantiphi","answer":"PD"}))



