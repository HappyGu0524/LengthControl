import torch
from datasets import load_dataset
from tqdm import tqdm
import json
import rouge_utils
import length_utils
import process_utils
import random
import numpy as np
import argparse
import logging
import os
import shutil
from openai import OpenAI
from count import word_count

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='gpt-4-turbo')
parser.add_argument('--task', type=str, default='cnndm')
parser.add_argument('--control_mode', type=str, default='off')
parser.add_argument('--api_keys', type=str)
parser.add_argument('--base_url', type=str, default=None)

parser.add_argument('--trials', type=int, default=5)

parser.add_argument('--resume_eval', action='store_true')


args = parser.parse_args()


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device='cuda:0'

PATH = "YOUR LOCAL DIR"

client = OpenAI(
    api_key=args.api_keys,
    base_url=args.base_url
)

if args.control_mode == 'off':
    rewrite = False
elif args.control_mode in ['on', 'rand', 'mh']:
    rewrite = True

log_dir = f"./outputs/{args.task}_{args.model_name}_{args.control_mode}/"
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

logging.info(f"Task: {args.task}")
logging.info(f"Model: {args.model_name}")
logging.info(f"Control: {args.control_mode}")
logging.info(f"Trials: {args.trials}")
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


for i in tqdm(range(0, 200)):
    if LINE_NUM is not None:
        if i < LINE_NUM:
            continue
        elif i >= LINE_NUM:
            logging.info(f"Start from example: {i}")
            example = dataset[i]
    else:
        example = dataset[i]

    if args.task == 'alpaca':
        instructions.append(example['instruction'])
        word_nums.append(example['words'])
    elif args.task == 'mtbench':
        instructions.append(example['instruction'])
        categories.append(example['category'])
        word_nums.append(example['words'])
        oracle_answers.append(example['oracle_answer'])
    
    example_info = process_utils.process_chat_api(example, task=process_task, num_control=True, icl=True, CASE=CNNDM_CASE)

    ref_words = example_info["words"]
    oracle_target = example_info["reference"]
    
    references.append(oracle_target)

    message = example_info['message']

    response = client.chat.completions.create(
        model=args.model_name,
        messages=message,
    )
    generated_target = response.choices[0].message.content
    if args.task == 'cnndm' and generated_target.startswith('Summary:'):
        generated_target = generated_target.removeprefix('Summary:').strip()

    


    if rewrite:
        pred_words = word_count(generated_target, args.task)
        if length_score(pred_words, ref_words) == 0:
            predictions.append(generated_target)
        else:
            if args.control_mode == 'rand':
                chosen_target = generated_target
                chosen_score = length_score(pred_words, ref_words)

                for ind_trial in range(args.trials):
                    response = client.chat.completions.create(
                        model=args.model_name,
                        messages=message,
                    )
                    candidate_target = response.choices[0].message.content
                    if args.task == 'cnndm' and generated_target.startswith('Summary:'):
                        candidate_target = candidate_target.removeprefix('Summary:').strip()

                    early_stop = False
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
                    input_dict['article'] = example['article']
                else:
                    input_dict['instruction'] = example['instruction']

                prediction_candidates = [(pred_words, generated_target)]
                independent_scores = [0]
                for ind_trial in range(args.trials):
                    random_index = random.randrange(len(prediction_candidates))
                    prev_wrds, prev_candidate = prediction_candidates.pop(random_index)
                    independent_scores.pop(random_index)
                        
                    input_dict['prediction'] = prev_candidate
                    input_dict['pred_words'] = prev_wrds
                        
                    message = process_utils.process_resample_api(input_dict, task=process_task, mode=args.control_mode)

                    response = client.chat.completions.create(
                        model=args.model_name,
                        messages=message,
                    )
                    current_candidate = response.choices[0].message.content
                    if args.task == 'cnndm' and generated_target.startswith('Summary:'):
                        current_candidate = current_candidate.removeprefix('Summary:').strip()
                    current_wrds = word_count(current_candidate.strip(), args.task)

                        #current - prev
                    eval_dict = {
                        'article': input_dict['article'] if args.task == 'cnndm' else None,
                        'instruction': input_dict['instruction'] if args.task in ['alpaca', 'mtbench'] else None,
                        'prediction_pair': [current_candidate, prev_candidate],
                        'words': ref_words,
                        'pred_words_pair': [current_wrds, prev_wrds],
                        'category': example_info['category'] if args.task == 'mtbench' else None
                    }
                    eval_score, independent_score = process_utils.process_score_api(eval_dict, client, model=args.model_name, length_score=length_score, task=process_task)
                    current_ind_score, prev_ind_score = independent_score

                        
                    early_stop = False

                    if length_score(current_wrds, ref_words) == 0:
                        early_stop = True
                            
                    if eval_score >= 1.0:
                        prediction_candidates.append((current_wrds, current_candidate))
                        independent_scores.append(current_ind_score)
                    else:
                        prediction_candidates.append((prev_wrds, prev_candidate))
                        independent_scores.append(prev_ind_score)
                        if random.random() < eval_score:
                            prediction_candidates.append((current_wrds, current_candidate))
                            independent_scores.append(current_ind_score)
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
        predictions.append(generated_target.strip())

    if args.task == 'cnndm':
        with open(log_dir+"references.out", "a") as file:
            file.write(' '.join(references[-1].strip().split()) + '\n')
            file.flush()

        with open(log_dir+"predictions.out", "a") as file:
            file.write(' '.join(predictions[-1].strip().split()) + '\n')
            file.flush()

    elif args.task == 'alpaca':
        with open(log_dir+"predictions.jsonl", "a") as file:
            file.write(json.dumps({'instruction': instructions[-1], 'prediction': predictions[-1], 'reference': references[-1], 'words': word_nums[-1]}) + '\n')
            file.flush()
    elif args.task == 'mtbench':
        with open(log_dir+"predictions.jsonl", "a") as file:
            file.write(json.dumps({'instruction': instructions[-1], 'prediction': predictions[-1], 'reference': references[-1], 'words': word_nums[-1], 'category': categories[-1], 'oracle_answer': oracle_answers[-1]}) + '\n')
            file.flush()


from functools import partial
num_count = partial(word_count, task=args.task)
length_results = length_utils.compute(predictions=predictions, references=references, length_score=length_score, num_count=num_count)


if args.task == 'cnndm':
    results = rouge_utils.compute(predictions=predictions,references=references)
    logging.info(results)
logging.info(length_results)

