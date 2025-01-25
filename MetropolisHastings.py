import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
import rouge_utils
import length_utils
import llmeval_utils
import process_utils
import random
import numpy as np
import copy
import math
import argparse
import logging
import os
import shutil
import gc
from count import word_count

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='llama3.1')
parser.add_argument('--task', type=str, default='cnndm')
parser.add_argument('--control_mode', type=str, default='off')

parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--beam', type=int, default=32)

parser.add_argument('--resume_eval', action='store_true')
parser.add_argument('--expandable_segments', action='store_true')
parser.add_argument('--dup', type=str, default=None)

args = parser.parse_args()

if args.expandable_segments:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device='cuda:0'

PATH = "YOUR LOCAL DIR"

if args.model_name == 'llama3.1':
    model_name = "Llama-3.1-8B-Instruct"
    gen_len = 1024
    eval_len = 1024
elif args.model_name == 'llama3':
    model_name = "Meta-Llama-3-8B-Instruct"
    gen_len = 1024
    eval_len = 1024
elif args.model_name == 'llama2':
    model_name = "Llama-2-7b-chat-hf"
    gen_len = 300
    eval_len = 300
elif args.model_name == 'llama3.2V':
    model_name = "Llama-3.2-11B-Vision-Instruct"
    gen_len = 1024
    eval_len = 1024
elif args.model_name == 'qwen2.5':
    model_name = "Qwen2.5-7B-Instruct"
    gen_len = 1024
    eval_len = 1024
checkpoint = PATH + model_name

tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        use_fast=False
    )
tokenizer.pad_token = tokenizer.eos_token
if args.model_name == 'qwen2.5':
    tokenizer.eos_token = "<|endoftext|>"

if args.control_mode == 'off':
    rewrite = False
elif args.control_mode in ['on', 'rand', 'mh']:
    rewrite = True

log_dir = f"./outputs/{args.task}_{args.model_name}_{args.control_mode}/"
if args.dup is not None:
    log_dir = f"./outputs/{args.dup}_{args.task}_{args.model_name}_{args.control_mode}/"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "out.log")



logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logging.info("Starting process with PID: %s", os.getpid())
logging.info(f"Task: {args.task}")
logging.info(f"Model: {args.model_name}")
logging.info(f"Control: {args.control_mode}")
logging.info(f"Batch Size: {args.batch_size}")
logging.info(f"Trials: {args.trials}")
logging.info(f"Beam Size: {args.beam}")
logging.info(f"Output Dir: {log_dir}")
logging.info(f"Log File: {log_file}")



CNNDM_CASE = [{
        'article': 'LONDON, England (CNN) -- French Foreign Minister Bernard Kouchner\'s declaration that France had to prepare for the possibility of war against Iran over its nuclear program was not conventional diplomatic behavior. But then Kouchner was never expected to be a soft-soaper on the diplomatic scene. French foreign minister Bernard Kouchner has a reputation for challenging convention and authority. A surprise appointment from the Socialist ranks to Nicolas Sarkozy\'s conservative government, the founder of Medicins Sans Frontiers has always challenged convention and authority. The former UN Secretary General Boutros Boutros-Ghali once called Kouchner \'an unguided missile\' and the man himself has been known to declare: "To change the law you sometimes have to break the law". He was in his youth one of the leaders of the students revolt in France in May 1968. Kouchner is a humanitarian as well as a patriot, with a strong commitment to human rights. Unusually for a man of the Left, he supported the US-led intervention in Iraq (while criticizing the aftermath). But he did so on the grounds of Saddam Hussein\'s denial of human rights, not his possible possession of weapons of mass destruction. His and President Sarkozy\'s concern for human rights lies behind their eagerness to join Gordon Brown\'s Britain in a new push for action in Darfur. Bernard Kouchner did not come to his position with any of former President Chirac\'s instinctive distrust of the United States. Washington, which has been critical of some European states for their weakness in confronting Teheran, will have been delighted by his \'get serious\' warning to Teheran. But the plain-speaking Kouchner is unlikely to be deterred by fears of upsetting the White House when he has criticisms to make of US policy. How much should be made of his words on Iran remains unclear at this stage. They were scarcely on the same scale as President Chirac\'s threat when he was still in office to retaliate with nuclear strikes against any state found to be responsible for a large-scale terrorist attack on France. But they are all of a piece with France\'s new high-profile style under the presidency of Nicolas Sarkozy. Mr Kouchner, for example, became the first French Foreign Minister to visit Iraq since 1988, insisting that there could only be a political solution to the country\'s problems, not a military one, and offering France\'s services as a mediator and \'honest broker\' between Sunnis, Shiites and Kurds. On Iran he is, in a way, merely echoing the words of his President who declared in a speech last month that a nuclear-armed Iran would be \'unacceptable\' and describing the stand-off over its nuclear program as \'undoubtedly the most serious crisis before us today\'. Certainly Mr Kouchner is making clear that France no longer takes the view once expressed by President Chirac that a nuclear-armed Iran might be inevitable . In continuing to ratchet up the rhetoric over that threat and to underline the West\'s resolution on Iran\'s nuclear enrichment program Mr Kouchner is supplementing his president\'s warnings. Neither is saying that military intervention against Iran is imminent or inevitable. Neither has yet confirmed that France would be part of any such military action. But both are stressing the risks which are piling up as a result of Teheran\'s brinkmanship. Perhaps the strongest lesson though from Mr Kouchner\'s intervention is his underlining that the new administration in France is not a knee-jerk anti-American one -- and that France is in the business of reclaiming a role at the top diplomatic tables. E-mail to a friend.',
        'highlights': 'French FM Kouchner has told France to prepare for possibility of war with Iran. Was a surprise appointment to Nicolas Sarkozy\'s conservative government. Also the first French Foreign Minister to visit Iraq since 1988. Founder of Medicins Sans Frontiers, also French student leader in May 1968.'
    }]
for content in CNNDM_CASE:
    content['words'] = len(content['highlights'].split())


if args.task == "cnndm":
    process_task = 'summary'

    dataset = load_dataset("json", data_files='./TestSet/cnndm.jsonl', split='train')

    def process_cnndm(example):
        sent = example["highlights"]
        sent = sent.replace(" .", ".")
        word_num = len(sent.split())
        example["highlights"] = sent
        example["words"] = word_num
        return example

    dataset = dataset.map(process_cnndm)

    def length_score(pred_words, ref_words):
        return abs(pred_words - ref_words)



elif args.task == "alpaca":
    process_task = 'alpaca'

    dataset = load_dataset("json", data_files='./TestSet/alpaca.jsonl', split='train')

    def process_alpaca(example):
        out_example = {}
        length_instruction = example['instruction']
        out_example['length_instruction'] = length_instruction
        out_example['instruction'] = '\n\n'.join(length_instruction.strip().split('\n\n')[1:])
        sent = example['output'].strip()
        word_num = word_count(sent, args.task)
        assert word_num == example['max_len']
        out_example["words"] = word_num
        out_example["reference"] = sent
        return out_example
    
    dataset = dataset.map(process_alpaca)

    def length_score(pred_words, ref_words):
        return max(0, (pred_words - ref_words))
    
elif args.task == "mtbench":
    process_task = 'mtbench'

    dataset = load_dataset("json", data_files='./TestSet/mtbench.jsonl', split='train')
    
    def process_mtbench(example):
        out_example = {}
        length_instruction = example['instruction']
        out_example['length_instruction'] = length_instruction
        out_example['instruction'] = '\n\n'.join(length_instruction.strip().split('\n\n')[1:])
        #print(out_example['instruction'])

        sent = example['reference'].strip()
        word_num = word_count(sent, args.task)
        assert word_num == example['max_len']
        out_example['words'] = word_num
        out_example['reference'] = sent
        out_example['category'] = example['category']
        out_example['oracle_answer'] = example['oracle_answer']
        return out_example

    dataset = dataset.map(process_mtbench)
    def length_score(pred_words, ref_words):
        return max(0, (pred_words - ref_words))




model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

references = []
predictions = []
instructions = []
categories = []
oracle_answers = []
word_nums = []

LINE_NUM = None
if args.task == 'cnndm':
    if args.resume_eval and os.path.exists(log_dir+"references.out") and os.path.exists(log_dir+"predictions.out"):
        shutil.copy(log_dir+"references.out", log_dir+"references.tmp")
        shutil.copy(log_dir+"predictions.out", log_dir+"predictions.tmp")
        with open(log_dir+"references.tmp", "r") as file:
            LINE_NUM = len(file.readlines())
        with open(log_dir+"predictions.tmp", "r") as file:
            LINE_NUM = min(LINE_NUM, len(file.readlines()))
        logging.info(f"Resume Evaluation from line {LINE_NUM}")
    else:
        with open(log_dir+"references.out", "w") as file:
            file.write("")
        with open(log_dir+"predictions.out", "w") as file:
            file.write("")

elif args.task in ['alpaca', 'mtbench']:
    if args.resume_eval and os.path.exists(log_dir+"predictions.jsonl"):
        shutil.copy(log_dir+"predictions.jsonl", log_dir+"predictions.jsonl.tmp")
        with open(log_dir+"predictions.jsonl.tmp", "r") as file:
            LINE_NUM = len(file.readlines())
    else:
        with open(log_dir+"predictions.jsonl", "w") as file:
            file.write("")



generation_config = model.generation_config
logging.info(generation_config)


for i in tqdm(range(0, len(dataset), args.batch_size)):
    if LINE_NUM is not None:
        if i + args.batch_size <= LINE_NUM:
            logging.info(f"Skip the batch {i}:{i+args.batch_size}")
            continue
        elif i < LINE_NUM and i + args.batch_size > LINE_NUM:
            batch = dataset[LINE_NUM: i + args.batch_size]
        elif i >= LINE_NUM:
            batch = dataset[i : i + args.batch_size]
    else:
        batch = dataset[i : i + args.batch_size]

    if args.task == 'alpaca':
        instructions.extend(batch['instruction'])
        word_nums.extend(batch['words'])
    elif args.task == 'mtbench':
        instructions.extend(batch['instruction'])
        categories.extend(batch['category'])
        word_nums.extend(batch['words'])
        oracle_answers.extend(batch['oracle_answer'])

    batch_info = process_utils.process_chat(batch, tokenizer, task=process_task, num_control=True, icl=True, CASE=CNNDM_CASE)
    input_ids = batch_info['input_ids']
    attention_mask = batch_info['attention_mask']


    batch_size, input_seq_len = input_ids.shape

    target_word_length = batch_info["words"]
    oracle_targets = batch_info["reference"]
    references.extend(oracle_targets)


    generated_ids = model.generate(input_ids.to(device), attention_mask=attention_mask.to(device), max_new_tokens=gen_len, use_cache=True).to('cpu')
    generated_targets = [tokenizer.decode(generated_ids[b,input_seq_len:], skip_special_tokens=True).strip() for b in range(batch_size)]
    
    #del input_ids
    #del attention_mask
    #del generated_ids
    #gc.collect()
    #torch.cuda.empty_cache()

    if rewrite:
        for ind in range(batch_size):
            ref_words = target_word_length[ind]
            generated_target = generated_targets[ind]
            pred_words = word_count(generated_target, args.task)


            if length_score(pred_words, ref_words) == 0:
                predictions.append(generated_target)
            else:
                if args.control_mode == 'rand':
                    chosen_target = generated_target
                    chosen_score = length_score(pred_words, ref_words)

                    for ind_trial in range(args.trials):
                        tmp_input_ids = input_ids[ind].clone().unsqueeze(0).repeat(args.beam, 1)
                        tmp_attention_mask = attention_mask[ind].clone().unsqueeze(0).repeat(args.beam, 1)

                        beam_size, input_seq_len = tmp_input_ids.shape
                        
                        candidate_ids = model.generate(tmp_input_ids.to(device), attention_mask=tmp_attention_mask.to(device), max_new_tokens=gen_len, use_cache=True).to('cpu')

                        early_stop = False
                        for b in range(beam_size):
                            candidate_target = tokenizer.decode(candidate_ids[b,input_seq_len:], skip_special_tokens=True).strip()
                            current_words = word_count(candidate_target, args.task)
                            if length_score(current_words, ref_words) == 0:
                                early_stop = True

                            if length_score(current_words, ref_words) < chosen_score:
                                chosen_target = candidate_target
                                chosen_score = length_score(current_words, ref_words)

                        if early_stop:
                            break

                    predictions.append(chosen_target.strip())

                else:
                    input_dict = {
                        "words": ref_words
                    }
                    if args.task == 'cnndm':
                        input_dict['article'] = batch["article"][ind]
                    else:
                        input_dict['instruction'] = batch['instruction'][ind]

                    prediction_candidates = [(pred_words, generated_target)]*args.beam
                    for ind_trial in range(args.trials):
                        scores = []
                        input_token_list = []
                        for score, candidate in prediction_candidates:
                            input_dict['prediction'] = candidate
                            input_dict['pred_words'] = word_count(candidate, args.task)
                            input_tokens = process_utils.process_resample(input_dict, tokenizer, task=process_task, mode=args.control_mode)
                            scores.append(score)
                            input_token_list.append(input_tokens)
                        
                        tokenization = tokenizer(input_token_list, padding=True, padding_side='left', add_special_tokens=False, return_tensors='pt')
                        input_ids, attention_mask = tokenization.input_ids, tokenization.attention_mask
                            
                        beam_size, input_seq_len = input_ids.shape

                        candidate_ids = model.generate(input_ids.to(device), attention_mask=attention_mask.to(device), max_new_tokens=gen_len, use_cache=True).to('cpu')
                        candidate_targets = [tokenizer.decode(candidate_ids[b,input_seq_len:], skip_special_tokens=True).strip() for b in range(beam_size)]

                        #del input_ids
                        #del attention_mask
                        #del candidate_ids
                        #gc.collect()
                        torch.cuda.empty_cache()

                        candidate_scores = [word_count(cnd, args.task) for cnd in candidate_targets]


                        eval_dict = []
                        for b in range(beam_size):
                            prev_words, prev_prediction = prediction_candidates[b]
                            current_words, current_prediction = candidate_scores[b], candidate_targets[b]
                            eval_dict.append(
                                {
                                    'article': input_dict['article'] if args.task == 'cnndm' else None,
                                    'instruction': input_dict['instruction'] if args.task in ['alpaca', 'mtbench'] else None,
                                    'prediction_pair': [current_prediction, prev_prediction],
                                    'words': ref_words,
                                    'pred_words_pair': [current_words, prev_words],
                                    'category': batch_info['category'][ind] if args.task == 'mtbench' else None
                                }
                            )
                        eval_scores, independent_scores = process_utils.process_score(eval_dict, tokenizer, model, length_score=length_score, task=process_task, max_new_tokens=eval_len, importance_factor=0)#######
                        
                        early_stop = False
                        for b in range(beam_size):
                            current_words, current_prediction = candidate_scores[b], candidate_targets[b]
                            if length_score(current_words, ref_words) == 0:
                                early_stop = True
                            
                            if eval_scores[b] >= 1.0:
                                prediction_candidates[b] = (current_words, current_prediction)
                            else:
                                if random.random() < eval_scores[b]:
                                    prediction_candidates[b] = (current_words, current_prediction)
                                else:
                                    pass
                        if early_stop:
                            break

                    chosen_score_l = None #minimum
                    chosen_score_q = None #maximum
                    chosen_target = None
                    for l in range(len(prediction_candidates)):
                        cand_wrds, candidate = prediction_candidates[l]
                        q_score = independent_scores[l]
                        if chosen_score_l is None or chosen_score_l > length_score(cand_wrds, ref_words):
                            chosen_score_l = length_score(cand_wrds, ref_words)
                            chosen_score_q = q_score
                            chosen_target = candidate
                        elif chosen_score_l == length_score(cand_wrds, ref_words) and chosen_score_q < q_score:
                            chosen_score_q = q_score
                            chosen_target = candidate

                    predictions.append(chosen_target.strip()) 

    else:
        predictions.extend(generated_targets)


    if args.task == 'cnndm':
        with open(log_dir+"references.out", "a") as file:
            for ref in references[-batch_size:]:
                file.write(' '.join(ref.strip().split()) + '\n')
            file.flush()

        with open(log_dir+"predictions.out", "a") as file:
            for pred in predictions[-batch_size:]:
                file.write(' '.join(pred.strip().split()) + '\n')
            file.flush()

    elif args.task == 'alpaca':
        with open(log_dir+"predictions.jsonl", "a") as file:
            for inst, pred, ref, wrd in zip(instructions[-batch_size:], predictions[-batch_size:], references[-batch_size:], word_nums[-batch_size:]):
                file.write(json.dumps({'instruction': inst, 'prediction': pred, 'reference': ref, 'words': wrd}) + '\n')
            file.flush()
    elif args.task == 'mtbench':
        with open(log_dir+"predictions.jsonl", "a") as file:
            for inst, pred, ref, wrd, cate, orans in zip(instructions[-batch_size:], predictions[-batch_size:], references[-batch_size:],  word_nums[-batch_size:], categories[-batch_size:], oracle_answers[-batch_size:]):
                file.write(json.dumps({'instruction': inst, 'prediction': pred, 'reference': ref, 'words': wrd, 'category': cate, 'oracle_answer': orans}) + '\n')
            file.flush()




from functools import partial
num_count = partial(word_count, task=args.task)
length_results = length_utils.compute(predictions=predictions, references=references, length_score=length_score, num_count=num_count)

if args.task == 'cnndm':
    results = rouge_utils.compute(predictions=predictions,references=references)

elif args.task == 'alpaca':
    results = llmeval_utils.compute(instructions, predictions, references, model, tokenizer, task='alpaca', word_count=num_count, max_words=word_nums, length_score=length_score)

elif args.task == 'mtbench':
    results = llmeval_utils.compute(instructions, predictions, references, model, tokenizer, task='mtbench', categories=categories, oracle_answers=oracle_answers, word_count=num_count, max_words=word_nums, length_score=length_score)

logging.info(results)

logging.info(length_results)
