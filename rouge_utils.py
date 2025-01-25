from rouge_score import rouge_scorer, scoring

def compute(predictions, references, rouge_types=None, use_aggregator=True, use_stemmer=False):
    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer)
    if use_aggregator:
        aggregator = scoring.BootstrapAggregator()
    else:
        scores = []

    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)
        if use_aggregator:
            aggregator.add_scores(score)
        else:
            scores.append(score)

    if use_aggregator:
        result = aggregator.aggregate()
        for key in result:
            result[key] = result[key].mid.fmeasure

    else:
        result = {}
        for key in scores[0]:
            result[key] = list(score[key].fmeasure for score in scores)

    return result


if __name__ == "__main__":
    import argparse

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
    with open(log_dir+"references.out", "r") as file:
        for line in file.readlines():
            references.append(line.strip())

    with open(log_dir+"predictions.out", "r") as file:
        for line in file.readlines():
            predictions.append(line.strip())
    if args.prefix is not None:
        prefix = args.prefix
    else:
        prefix = len(predictions)
    print(compute(predictions=predictions[:prefix],references=references[:prefix]))