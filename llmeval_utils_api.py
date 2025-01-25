import re
from itertools import zip_longest
import numpy as np
from tqdm import tqdm
def template(instruction, prediction, reference, task='alpaca', category=None, oracle_answer=None):
    message = []
    if task == 'alpaca':
        system_prompt = "You are a highly efficient assistant, who evaluates and selects the best large language model (LLMs) based on the quality of their responses to a given instruction. "\
                        "This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers."
        if system_prompt is not None:
            message.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        user_message = "I require a leaderboard for various large language models. "\
            "I'll provide you with prompts given to these models and their corresponding outputs. "\
            "Your task is to assess these responses, and select the model that produces the best output from a human perspective.\n\n"\
            "## Instruction\n\n"\
            f'{{\n"instruction": """{instruction}""",\n}}\n\n'\
            "## Model Outputs\n\n"\
            "Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.\n\n"\
            f'{{\n    {{\n        "model_identifier": "m",\n        "output": """{prediction}"""\n    }},\n    {{\n        "model_identifier": "M",\n        "output": """{reference}"""\n    }}\n}}\n\n'\
            "## Task\n\n"\
            "Evaluate the models based on the quality and relevance of their outputs, and select the model that generated the best output. "\
            "Answer by providing the model identifier of the best model. We will use your output as the name of the best model, "\
            "so make sure your output only contains one of the following model identifiers and nothing else (no quotes, no spaces, no new lines, ...): \"[[m]]\" or \"[[M]]\".\n\n"\
            "## Best Model Identifier\n"
        message.append({"role": "user", "content": [{"type": "text", "text": user_message}]})
    elif task == 'mtbench':
        assert category is not None
        assert oracle_answer is not None
        if category in ['reasoning', 'math', 'coding']:
            assert oracle_answer != 'none'
            system_prompt = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. "\
                "Your evaluation should consider correctness and helpfulness. You will be given a reference answer, assistant A's answer, and assistant B's answer. "\
                "Your job is to evaluate which assistant's answer is better. Begin your evaluation by comparing both assistants' answers with the reference answer. "\
                "Identify and correct any mistakes. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. "\
                "Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. "\
                "Be as objective as possible. After providing your brief explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."
            if system_prompt is not None:
                message.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

            user_message = f"[User Question]\n{instruction}\n\n[The Start of Reference Answer]\n{oracle_answer}\n[The End of Reference Answer]\n\n[The Start of Assistant A's Answer]\n{prediction}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{reference}\n[The End of Assistant B's Answer]\n\n"
            message.append({"role": "user", "content": [{"type": "text", "text": user_message}]})

        else:
            system_prompt = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. "\
                "You should choose the assistant that follows the user's instructions and answers the user's question better. "\
                "Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. "\
                "Begin your evaluation by comparing the two responses and provide a short explanation. "\
                "Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. "\
                "Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. "\
                "Be as objective as possible. After providing your brief explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."
            if system_prompt is not None:
                message.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
            
            user_message = f"[User Question]\n{instruction}\n\n[The Start of Assistant A's Answer]\n{prediction}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{reference}\n[The End of Assistant B's Answer]\n\n"
            message.append({"role": "user", "content": [{"type": "text", "text": user_message}]})

    return message

def compute(instructions, predictions, references, client, eval_model, task='alpaca', categories=None, oracle_answers=None, word_count=None, max_words=None, length_score=None):
    result = []
    if task == 'alpaca':
        pattern = r"\[\[m\]\]|\[\[M\]\]"
        for instruction, prediction, reference, word in tqdm(zip_longest(instructions, predictions, references, max_words or [], fillvalue=None)):
            message = template(instruction, prediction, reference, task=task)
            response = client.chat.completions.create(
                model=eval_model,
                messages=message,
            )
            output = response.choices[0].message.content

            try:
                match = re.findall(pattern, output)[0]
                if match == "[[m]]":
                    ans = 1.0
                elif match == "[[M]]":
                    ans = 0.0
                else:
                    raise Exception
                if word_count is not None and word is not None and length_score is not None:
                    if length_score(word_count(prediction), word) > 0:
                        ans = 0.0
            
                result.append(ans)
            except Exception:
                print('unable to get the score')
                continue

    elif task == 'mtbench':
        assert categories is not None
        pattern = r"\[\[A\]\]|\[\[B\]\]|\[\[C\]\]"
        for instruction, prediction, reference, category, oracle_answer, word in tqdm(zip_longest(instructions, predictions, references, categories, oracle_answers, max_words or [], fillvalue=None)):
            message = template(instruction, prediction, reference, task=task, category=category, oracle_answer=oracle_answer)
            response = client.chat.completions.create(
                model=eval_model,
                messages=message,
            )
            output = response.choices[0].message.content

            try:
                match = re.findall(pattern, output)[0]
                if match == "[[A]]":
                    ans = 1.0
                elif match == "[[B]]":
                    ans = 0.0
                elif match == "[[C]]":
                    ans = 0.5
                else:
                    raise Exception
                if word_count is not None and word is not None and length_score is not None:
                    if length_score(word_count(prediction), word) > 0:
                        ans = 0.0
                result.append(ans)
            except Exception:
                print('unable to get the score')
                continue

    return np.mean(result)



    



if __name__ == "__main__":
    import argparse
    import json
    from count import word_count
    from functools import partial
    from openai import OpenAI
    import torch

    def none_or_int(value):
        if value.lower() == "none":
            return None
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid int or None value: {value}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3.1')
    parser.add_argument('--eval_model', type=str, default='gpt-4o')
    parser.add_argument('--task', type=str, default='alpaca')
    parser.add_argument('--control_mode', type=str, default='off')
    parser.add_argument('--prefix', type=none_or_int, default=None)
    parser.add_argument('--api_keys', type=str)
    parser.add_argument('--base_url', type=str, default=None)
    args = parser.parse_args()
    log_dir = f"./outputs/{args.task}_{args.model_name}_{args.control_mode}/"

    def length_score(pred_words, ref_words):
        return max(0, (pred_words - ref_words))
    


    client = OpenAI(
        api_key=args.api_keys,
        base_url=args.base_url
    )
    model = args.eval_model


    if 'alpaca' in args.task:
        pred_dict = {
            'instructions': [],
            'predictions': [],
            'references': [],
            'max_words': [],
            'task': 'alpaca',
            'word_count': partial(word_count, task='alpaca'),
            'length_score': length_score,
            'eval_model': model,
            'client': client
        }
    elif 'mtbench' in args.task:
        pred_dict = {
            'instructions': [],
            'predictions': [],
            'references': [],
            'max_words': [],
            'categories': [],
            'oracle_answers': [],
            'task': 'mtbench',
            'word_count': partial(word_count, task='mtbench'),
            'length_score': length_score,
            'eval_model': model,
            'client': client
        }
    else:
        raise Exception

    with open(log_dir+"predictions.jsonl", "r") as file:
        for line in file.readlines() if args.prefix is None else file.readlines()[:args.prefix]:
            content_dict = json.loads(line.strip())
            pred_dict['instructions'].append(content_dict['instruction'])
            pred_dict['predictions'].append(content_dict['prediction'])
            pred_dict['references'].append(content_dict['reference'])
            pred_dict['max_words'].append(content_dict['words'])
            if 'mtbench' in args.task:
                pred_dict['categories'].append(content_dict['category'])
                pred_dict['oracle_answers'].append(content_dict['oracle_answer'])

            
    print(compute(**pred_dict))