import numpy as np
from count import word_count

def compute(predictions, references, length_score, num_count, length_types=None):
    if length_types is None:
        length_types = ["SQRT", "L1", "L2", "ACC"]
    pred_words = np.array([num_count(pred.strip()) for pred in predictions])
    ref_words = np.array([num_count(ref.strip()) for ref in references])

    output = {}
    if "SQRT" in length_types:
        output['SQRT'] = np.sqrt([length_score(pred_word, ref_word) for pred_word, ref_word in zip(pred_words, ref_words)]).mean()
    
    if "L1" in length_types:
        output['L1'] = np.mean([length_score(pred_word, ref_word) for pred_word, ref_word in zip(pred_words, ref_words)])
    
    if "L2" in length_types:
        output['L2'] = np.sqrt((np.array([length_score(pred_word, ref_word) for pred_word, ref_word in zip(pred_words, ref_words)]) ** 2).mean())

    if "ACC" in length_types:
        output['ACC'] = np.mean(np.array([length_score(pred_word, ref_word) for pred_word, ref_word in zip(pred_words, ref_words)]) == 0)
    #if "COS" in length_types:
    #    output['COS'] = (pred_words * ref_words).sum() / (np.sqrt((pred_words ** 2).sum()) * np.sqrt((ref_words ** 2).sum()))
    
    return output


if __name__ == "__main__":
    import argparse
    from functools import partial

    def none_or_int(value):
        if value.lower() == "none":
            return None
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid int or None value: {value}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3.1')
    parser.add_argument('--task', type=str, default='cnndm')
    parser.add_argument('--control_mode', type=str, default='off')
    parser.add_argument('--prefix', type=none_or_int, default=None)
    args = parser.parse_args()
    log_dir = f"./outputs/{args.task}_{args.model_name}_{args.control_mode}/"
    references = []
    predictions = []
    if 'cnndm' in args.task:
        with open(log_dir+"references.out", "r") as file:
            for line in file.readlines():
                references.append(line.strip())

        with open(log_dir+"predictions.out", "r") as file:
            for line in file.readlines():
                predictions.append(line.strip())
        
        def length_score(pred_words, ref_words):
            return abs(pred_words - ref_words)
        num_count = partial(word_count, task='cnndm')
    elif any(sub in args.task for sub in ['alpaca', 'mtbench']):
        import json
        with open(log_dir+"predictions.jsonl", 'r') as file:
            for line in file.readlines():
                content = json.loads(line.strip())
                references.append(content['reference'])
                predictions.append(content['prediction'])
        
        def length_score(pred_words, ref_words):
            return max(0, (pred_words - ref_words))

        num_count = partial(word_count, task='alpaca')

    if args.prefix is not None:
        prefix = args.prefix
    else:
        prefix = len(predictions)
    
    
    
    print(compute(predictions=predictions[:prefix],references=references[:prefix], length_score=length_score, num_count=num_count))