import re
import torch
import gc

def process_chat(batch, tokenizer, task='summary', icl=True, CASE=None, num_control=False):
    output = {}
    if task == 'summary':
        output['words'] = batch['words']
        output['reference'] = batch['highlights']
        articles = batch["article"]
        if icl and CASE is None:
            raise Exception

        system_prompt = "You are a powerful abstractive summarizer."
        tokenized_chats = []
        if num_control:
            for article, wrdnum in zip(articles, output['words']):
                message = []
                if system_prompt is not None:
                    message.append({"role": "system", "content":system_prompt})
                
                if icl:
                    user_message = f"Document:\n{CASE[0]['article']}\n\nBased on the previous document, provide a high-quality summary in exactly {CASE[0]['words']} words:"
                    message.append({"role":"user", "content":user_message})
                    assistant_message = f"Summary:\n{CASE[0]['highlights']}"
                    message.append({"role":"assistant", "content":assistant_message})
                    
                else:
                    pass
                    
                
                content_message = f"Document:\n{article}\n\nBased on the previous document, provide a high-quality summary in exactly {wrdnum} words:"
                message.append({"role":"user", "content":content_message})

                message.append({"role":"assistant", "content": "Summary:\n"})
                tokenized_chat = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False, continue_final_message=True)
                tokenized_chats.append(tokenized_chat)


        else:
            for article in articles:
                message = []
                if system_prompt is not None:
                    message.append({"role": "system", "content":system_prompt})
                
                if icl:
                    
                    user_message = f"Document:\n{CASE[0]['article']}\n\nBased on the previous document, provide a high-quality summary:"
                    message.append({"role":"user", "content":user_message})
                    assistant_message = f"Summary:\n{CASE[0]['highlights']}"
                    message.append({"role":"assistant", "content":assistant_message})
                else:
                    pass
                    
                content_message = f"Document:\n{article}\n\nBased on the previous document, provide a high-quality summary:"
                message.append({"role":"user", "content":content_message})

                message.append({"role":"assistant", "content": "Summary:\n"})
                tokenized_chat = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False, continue_final_message=True)
                tokenized_chats.append(tokenized_chat)



    elif task in ['alpaca', 'mtbench']:
        output['words'] = batch['words']
        output['reference'] = batch['reference']
        if task == 'mtbench':
            output['category'] = batch['category']
        
        system_prompt = None
        icl = False
        CASE = None
        tokenized_chats = []

        if num_control:
            instructions = batch['length_instruction']
        else:
            instructions = batch['instruction']
        
        for instruction in instructions:
            message = []
            if system_prompt is not None:
                message.append({"role": "system", "content":system_prompt})
            
            content_message = f"{instruction}"
            message.append({"role":"user", "content":content_message})
            tokenized_chat = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            tokenized_chats.append(tokenized_chat)


    tokenization = tokenizer(tokenized_chats, padding=True, padding_side='left', add_special_tokens=False, return_tensors='pt')

    input_ids, attention_mask = tokenization.input_ids, tokenization.attention_mask

    output['input_ids'] = input_ids
    output['attention_mask'] = attention_mask

    return output


def process_chat_api(example, task='summary', icl=True, CASE=None, num_control=False):
    output = {}
    if task == 'summary':
        output['words'] = example['words']
        output['reference'] = example['highlights']
        article = example["article"]
        if icl and CASE is None:
            raise Exception

        system_prompt = "You are a powerful abstractive summarizer."
        if num_control:
            wrdnum = output['words']
            
            message = []
            if system_prompt is not None:
                message.append({"role": "system", "content":[{"type": "text","text": system_prompt}]})
                
            if icl:
                user_message = f"Document:\n{CASE[0]['article']}\n\nBased on the previous document, provide a high-quality summary in exactly {CASE[0]['words']} words:"
                message.append({"role":"user", "content":[{"type": "text", "text": user_message}]})
                assistant_message = f"Summary:\n{CASE[0]['highlights']}"
                message.append({"role":"assistant", "content":[{"type": "text", "text": assistant_message}]})
            else:
                pass
              
            content_message = f"Document:\n{article}\n\nBased on the previous document, provide a high-quality summary in exactly {wrdnum} words:"
            message.append({"role":"user", "content":[{"type": "text", "text": content_message}]})

            message.append({"role":"assistant", "content": [{"type": "text", "text": "Summary:\n"}]})

        else:
            message = []
            if system_prompt is not None:
                message.append({"role": "system", "content":[{"type": "text","text": system_prompt}]})
                
            if icl:
                user_message = f"Document:\n{CASE[0]['article']}\n\nBased on the previous document, provide a high-quality summary:"
                message.append({"role":"user", "content":[{"type": "text", "text": user_message}]})
                assistant_message = f"Summary:\n{CASE[0]['highlights']}"
                message.append({"role":"assistant", "content":[{"type": "text", "text": assistant_message}]})
            else:
                pass

            content_message = f"Document:\n{article}\n\nBased on the previous document, provide a high-quality summary:"
            message.append({"role":"user", "content":[{"type": "text", "text": content_message}]})

            message.append({"role":"assistant", "content": [{"type": "text", "text": "Summary:\n"}]})

        output['message'] = message


    elif task in ['alpaca', 'mtbench']:
        output['words'] = example['words']
        output['reference'] = example['reference']
        if task == 'mtbench':
            output['category'] = example['category']

        system_prompt = None
        icl = False
        CASE = None

        if num_control:
            instruction = example['length_instruction']
        else:
            instruction = example['instruction']

        message = []
        if system_prompt is not None:
            message.append({"role": "system", "content":[{"type": "text","text": system_prompt}]})
        
        content_message = f"{instruction}"
        message.append({"role":"user", "content":[{"type": "text", "text": content_message}]})
        
        output['message'] = message
    return output



def process_resample(input_dict, tokenizer, task='summary', mode='on'):
    #mode ["on", "mh"]
    if task == 'summary':
        article = input_dict['article']
        prediction = input_dict['prediction']
        words = input_dict['words']
        pred_words = input_dict['pred_words']

        message = []
        system_prompt = "You are a powerful abstractive summarizer."
        if system_prompt is not None:
            message.append({"role": "system", "content":system_prompt})
        
        user_message = f"Document:\n{article}\n\nBased on the previous document, provide a high-quality summary in exactly {words} words:"
        message.append({"role":"user", "content":user_message})
        assistant_message = f"Summary:\n{prediction}"
        message.append({"role":"assistant", "content":assistant_message})

        if mode == 'on':
            if pred_words > words:
                if pred_words - words > 3:
                    user_message = f"The generated summary is {'too ' if (pred_words - words > 10) else 'a bit '}long at {pred_words} words. Please improve it to be exactly {words} words by focusing on the core ideas and {'slightly ' if (pred_words - words <= 10) else ''} removing some redundant details:"
                else:
                    user_message = f"Please delete {pred_words - words} words appropriately based on the previous summary:"
            elif pred_words < words:
                if words - pred_words > 3:
                    user_message = f"The generated summary is {'too ' if (words - pred_words > 10) else 'a bit '}short at {pred_words} words. Please improve it to be exactly {words} words by {'slightly ' if (words - pred_words <= 10) else ''} adding some details and maintaining clarity and relevance:"
                else:
                    user_message = f"Please add {words - pred_words} words appropriately based on the previous summary:"
            else:
                user_message = f"The generated summary exactly meets the requirements at {pred_words} words. Please keep it to be exactly {words} words and improving the information coverage and conciseness:"
        elif mode == 'mh':
            user_message = f"Please generate an improved summary based on the previous one:"
        else:
            raise Exception

        message.append({"role":"user", "content":user_message})
        message.append({"role":"assistant", "content":"Summary:\n"})
        
        input_tokens = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False, continue_final_message=True)
    
    elif task in ['alpaca', 'mtbench']:
        instruction = input_dict['instruction']
        prediction = input_dict['prediction']
        words = input_dict['words']
        pred_words = input_dict['pred_words']

        message = []
        system_prompt = None
        if system_prompt is not None:
            message.append({"role": "system", "content":system_prompt})

        user_message = f"Answer the following instruction using {words} words or less.\n\n{instruction}"
        message.append({"role":"user", "content":user_message})
        assistant_message = f"Answer:\n{prediction}"
        message.append({"role":"assistant", "content":assistant_message})

        if mode == 'on':
            if pred_words > words:
                if pred_words - words > 3:
                    user_message = f"The generated answer is {'too ' if (pred_words - words > 10) else 'a bit '}long at {pred_words} words. Please improve it to be exactly {words} words or less by focusing on the core contents and {'slightly ' if (pred_words - words <= 10) else ''} removing any unhelpful, irrelevant, or inaccurate parts:"
                else:
                    user_message = f"Please delete {pred_words - words} words appropriately based on the previous response:"
            else:
                user_message = f"The generated answer exactly meets the requirements at {pred_words} words. Please keep it to be exactly {words} words or less and improving the quality, helpfulness, and relevance of the answer:"
        elif mode == 'mh':
            user_message = f"Please generate an improved answer based on the previous one:"
        else:
            raise Exception
        
        message.append({"role":"user", "content":user_message})
        message.append({"role":"assistant", "content":"Answer:\n"})
        input_tokens = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False, continue_final_message=True)

    return input_tokens#input_ids


def process_resample_api(input_dict, task='summary', mode='on'):
    #mode ['on', 'mh']
    if task == 'summary':
        article = input_dict['article']
        prediction = input_dict['prediction']
        words = input_dict['words']
        pred_words = input_dict['pred_words']

        message = []
        system_prompt = "You are a powerful abstractive summarizer."
        if system_prompt is not None:
            message.append({"role": "system", "content":[{"type": "text", "text": system_prompt}]})
        
        user_message = f"Document:\n{article}\n\nBased on the previous document, provide a high-quality summary in exactly {words} words:"
        message.append({"role":"user", "content": [{"type": "text", "text": user_message}]})
        assistant_message = f"Summary:\n{prediction}"
        message.append({"role":"assistant", "content": [{"type": "text", "text":assistant_message}]})

        if mode == 'on':
            if pred_words > words:
                if pred_words - words > 3:
                    user_message = f"The generated summary is {'too ' if (pred_words - words > 10) else 'a bit '}long at {pred_words} words. Please improve it to be exactly {words} words by focusing on the core ideas and {'slightly ' if (pred_words - words <= 10) else ''} removing some redundant details:"
                else:
                    user_message = f"Please delete {pred_words - words} words appropriately based on the previous summary:"
            elif pred_words < words:
                if words - pred_words > 3:
                    user_message = f"The generated summary is {'too ' if (words - pred_words > 10) else 'a bit '}short at {pred_words} words. Please improve it to be exactly {words} words by {'slightly ' if (words - pred_words <= 10) else ''} adding some details and maintaining clarity and relevance:"
                else:
                    user_message = f"Please add {words - pred_words} words appropriately based on the previous summary:"
            else:
                user_message = f"The generated summary exactly meets the requirements at {pred_words} words. Please keep it to be exactly {words} words and improving the information coverage and conciseness:"
                #raise Exception("`pred_words` equals `words`")
        elif mode == 'mh':
            user_message = f"Please generate an improved summary based on the previous one:"
        else:
            raise Exception
        
        message.append({"role":"user", "content": [{"type": "text", "text": user_message}]})
        message.append({"role":"assistant", "content": [{"type": "text", "text": "Summary:\n"}]})
    
    elif task in ['alpaca', 'mtbench']:
        instruction = input_dict['instruction']
        prediction = input_dict['prediction']
        words = input_dict['words']
        pred_words = input_dict['pred_words']

        message = []
        system_prompt = None
        if system_prompt is not None:
            message.append({"role": "system", "content":[{"type": "text", "text": system_prompt}]})

        user_message = f"Answer the following instruction using {words} words or less.\n\n{instruction}"
        message.append({"role":"user", "content": [{"type": "text", "text": user_message}]})
        assistant_message = f"Answer:\n{prediction}"
        message.append({"role":"assistant", "content": [{"type": "text", "text":assistant_message}]})

        if mode == 'on':
            if pred_words > words:
                if pred_words - words > 3:
                    user_message = f"The generated answer is {'too ' if (pred_words - words > 10) else 'a bit '}long at {pred_words} words. Please improve it to be exactly {words} words or less by focusing on the core contents and {'slightly ' if (pred_words - words <= 10) else ''} removing any unhelpful, irrelevant, or inaccurate parts:"
                else:
                    user_message = f"Please delete {pred_words - words} words appropriately based on the previous response:"
            else:
                user_message = f"The generated answer exactly meets the requirements at {pred_words} words. Please keep it to be exactly {words} words or less and improving the quality, helpfulness, and relevance of the answer:"
        elif mode == 'mh':
            user_message = f"Please generate an improved answer based on the previous one:"
        else:
            raise Exception
        
        message.append({"role":"user", "content": [{"type": "text", "text": user_message}]})
        message.append({"role":"assistant", "content": [{"type": "text", "text": "Answer:\n"}]})
        
    return message


def process_score(input_dict, tokenizer, model, length_score, task='summary', importance_factor=1.0, max_new_tokens=1024):
    #1 current 
    #2 previous
    if isinstance(input_dict, dict):
        prediction1, prediction2 = input_dict['prediction_pair']
        prediction1, prediction2 = [prediction1], [prediction2]
        words = [input_dict['words']]
        pred_words1, pred_words2 = input_dict['pred_words_pair']
        pred_words1, pred_words2 = [pred_words1], [pred_words2]
        article = [input_dict['article']]
        instruction = [input_dict['instruction']]
        category = [input_dict['category']]
    elif isinstance(input_dict, list):
        article, instruction, prediction1, prediction2, words, pred_words1, pred_words2, category = [], [], [], [], [], [], [], []
        for tmp_dict in input_dict:
            article.append(tmp_dict['article'])
            prediction1.append(tmp_dict['prediction_pair'][0])
            prediction2.append(tmp_dict['prediction_pair'][1])
            words.append(tmp_dict['words'])
            pred_words1.append(tmp_dict['pred_words_pair'][0])
            pred_words2.append(tmp_dict['pred_words_pair'][1])
            instruction.append(tmp_dict['instruction'])
            category.append(tmp_dict['category'])

    if task == 'summary':
        assert article[0] is not None

        tokens = []
        for i in range(len(article)):
            message = []
            system_prompt = "You are a powerful evaluator for abstractive summarization."
            if system_prompt is not None:
                message.append({"role": "system", "content":system_prompt})
            user_message = "I need to compare and evaluate the quality of two summaries generated for a given document. "\
                "Please provide a quantitative assessment of their performance based on the criteria below.\n\n"\
                f"Document:\n{article[i]}\n\nSummary 1:\n{prediction1[i]}\n\nSummary 2:\n{prediction2[i]}\n\n"\
                "Evaluation Criteria (each scored on a scale of 1-10, with 10 being the best):\n"\
                "1. Information Coverage: Does the summary include the most important and critical information from the document?\n"\
                "2. Linguistic Fluency: Are the sentences in the summary fluent, natural, and grammatically correct?\n"\
                "3. Conciseness: Does the summary avoid redundancy while retaining key information?\n"\
                "4. Logical Coherence: Is the summary well-structured with clear and logical flow?\n"\
                "5. Faithfulness: Does the summary accurately reflect the facts in the original document without adding false or misleading information?\n\n"\
                "Instructions:\n"\
                "* Score each summary based on the above criteria.\n"\
                "* Calculate an overall score for each summary as the sum of all criteria scores (maximum 50).\n"\
                "* Conclude by identifying which summary is better overall.\n"\
                "* Calculate a score ratio of Summary 1 to Summary 2 (Summary 1 Score ÷ Summary 2 Score).\n\n"\
                "Output Format:\n"\
                "#### Summary 1:\n"\
                "1. Information Coverage: [Score]/10\n"\
                "2. Linguistic Fluency: [Score]/10\n"\
                "3. Conciseness: [Score]/10\n"\
                "4. Logical Coherence: [Score]/10\n"\
                "5. Faithfulness: [Score]/10\n"\
                "**Overall Score:** [Total Score]/50\n\n"\
                "#### Summary 2:\n"\
                "1. Information Coverage: [Score]/10\n"\
                "2. Linguistic Fluency: [Score]/10\n"\
                "3. Conciseness: [Score]/10\n"\
                "4. Logical Coherence: [Score]/10\n"\
                "5. Faithfulness: [Score]/10\n"\
                "**Overall Score:** [Total Score]/50\n\n"\
                "### Conclusion:\n"\
                "- **Better Summary:** [Summary 1/Summary 2].\n"\
                "- **Score Ratio (Summary 1 ÷ Summary 2):** [Ratio, rounded to two decimal places].\n"
        
            message.append({"role":"user", "content":user_message})
            token = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            tokens.append(token)
        
        tokenization = tokenizer(tokens, padding=True, padding_side='left', add_special_tokens=False, return_tensors='pt')
        input_ids, attention_mask = tokenization.input_ids, tokenization.attention_mask
        batch_size, input_length = input_ids.shape

        output = model.generate(input_ids.to(model.device), attention_mask=attention_mask.to(model.device), max_new_tokens=max_new_tokens, use_cache=True).to('cpu')
        #del input_ids
        #del attention_mask
        #gc.collect()
        #torch.cuda.empty_cache()
        generations = [tokenizer.decode(output[b, input_length:], skip_special_tokens=True) for b in range(batch_size)]

        summary1_score_pattern = r"Summary 1:\s.*?Overall Score:\**\s(\d+)/50"
        summary2_score_pattern = r"Summary 2:\s.*?Overall Score:\**\s(\d+)/50"
        ratio_pattern = r"Score Ratio \(Summary 1 ÷ Summary 2\):\**\s(\d+.\d+)"

        final_scores = []
        independ_scores = []
        for i in range(len(generations)):
            tmp_gen = generations[i]
            summary1_score = re.search(summary1_score_pattern, tmp_gen, re.DOTALL)
            if summary1_score:
                try:
                    summary1_score = int(summary1_score.group(1))
                except Exception:
                    summary1_score = None
            else:
                summary1_score = None


            summary2_score = re.search(summary2_score_pattern, tmp_gen, re.DOTALL)
            if summary2_score:
                try:
                    summary2_score = int(summary2_score.group(1))
                except Exception:
                    summary2_score = None
            else:
                summary2_score = None

            ratio = re.search(ratio_pattern, tmp_gen)
            if ratio:
                try:
                    ratio = float(ratio.group(1))
                except Exception:
                    ratio = None
            else:
                ratio = None
        

            #quality_score > 1 means better current with higher quality
            if summary1_score is not None and summary2_score is not None:
                if summary2_score == 0:
                    if ratio is not None:
                        quality_score = ratio
                    else:
                        quality_score = 50
                else:
                    quality_score = summary1_score / summary2_score
            elif ratio is not None:
                quality_score = ratio
            else:
                print("Warning: can not get the quality score!!!")
                quality_score = 1.0
        

            #length_num > 1 means better current with lower error
            if length_score(pred_words2[i], words[i]) == 0 and length_score(pred_words1[i], words[i]) == 0:
                length_num = 1.0
            elif length_score(pred_words2[i], words[i]) == 0:
                length_num = 0.1
            elif length_score(pred_words1[i], words[i]) == 0:
                length_num = 10
            else:
                length_num = length_score(pred_words2[i], words[i]) / length_score(pred_words1[i], words[i])

            final_score = length_num * (quality_score ** importance_factor)
            final_scores.append(final_score)



            if summary1_score is not None:
                ind_q_score = summary1_score / 50
            else:
                ind_q_score = 0.8
            independ_scores.append(ind_q_score)



    elif task == 'alpaca':
        assert instruction[0] is not None

        tokens = []
        for i in range(len(instruction)):
            message = []
            system_prompt = "You are a highly efficient assistant, who evaluates and selects the best large language model (LLMs) based on the quality of their responses to a given instruction. "\
                            "This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers."
            if system_prompt is not None:
                message.append({"role": "system", "content":system_prompt})
            user_message = "I require a leaderboard for various large language models. "\
                "I'll provide you with an instruction given to these models and their corresponding responses. "\
                "Your task is to assess these responses, provide a quantitative assessment of their performance based on the criteria below, and select the model that produces the best output from a human perspective. "\
                "Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.\n\n"\
                f"Instruction:\n{instruction[i]}\n\nResponse 1:\n{prediction1[i]}\n\nResponse 2:\n{prediction2[i]}\n\n"\
                "Evaluation Criteria (each scored on a scale of 1-10, with 10 being the best):\n"\
                "1. Helpfulness: Does the response directly address the instruction and provide meaningful assistance?\n"\
                "2. Relevance: Does the response stay on topic and avoid unnecessary or unrelated information?\n"\
                "3. Accuracy: Is the information in the response factually correct and free of errors?\n"\
                "4. Depth: Does the response demonstrate a deep understanding of the topic, including nuanced explanations where relevant?\n"\
                "5. Creativity: Does the response display originality, creativity, or a unique approach to addressing the instruction?\n"\
                "6. Level of Detail: Is the response sufficiently detailed, providing comprehensive and thorough explanations where necessary?\n\n"\
                "Tasks:\n"\
                "* Score each response based on the above criteria.\n"\
                "* Calculate an overall score for each response as the sum of all criteria scores (maximum 60).\n"\
                "* Conclude by identifying which response is better overall.\n"\
                "* Calculate a score ratio of Response 1 to Response 2 (Response 1 Score ÷ Response 2 Score).\n\n"\
                "Output Format:\n"\
                "#### Response 1:\n"\
                "1. Helpfulness: [Score]/10\n"\
                "2. Relevance: [Score]/10\n"\
                "3. Accuracy: [Score]/10\n"\
                "4. Depth: [Score]/10\n"\
                "5. Creativity: [Score]/10\n"\
                "6. Level of Detail: [Score]/10\n"\
                "**Overall Score:** [Total Score]/60\n\n"\
                "#### Response 2:\n"\
                "1. Helpfulness: [Score]/10\n"\
                "2. Relevance: [Score]/10\n"\
                "3. Accuracy: [Score]/10\n"\
                "4. Depth: [Score]/10\n"\
                "5. Creativity: [Score]/10\n"\
                "6. Level of Detail: [Score]/10\n"\
                "**Overall Score:** [Total Score]/60\n\n"\
                "### Conclusion:\n"\
                "- **Better Response:** [Response 1/Response 2].\n"\
                "- **Score Ratio (Response 1 ÷ Response 2):** [Ratio, rounded to two decimal places].\n"
        
            message.append({"role":"user", "content":user_message})
            token = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            tokens.append(token)
        
        tokenization = tokenizer(tokens, padding=True, padding_side='left', add_special_tokens=False, return_tensors='pt')
        input_ids, attention_mask = tokenization.input_ids, tokenization.attention_mask
        batch_size, input_length = input_ids.shape

        output = model.generate(input_ids.to(model.device), attention_mask=attention_mask.to(model.device), max_new_tokens=max_new_tokens, use_cache=True).to('cpu')

        generations = [tokenizer.decode(output[b, input_length:], skip_special_tokens=True) for b in range(batch_size)]

        response1_score_pattern = r"Response 1:\s.*?Overall Score:\**\s(\d+)/60"
        response2_score_pattern = r"Response 2:\s.*?Overall Score:\**\s(\d+)/60"
        ratio_pattern = r"Score Ratio \(Response 1 ÷ Response 2\):\**\s(\d+.\d+)"

        final_scores = []
        independ_scores = []
        for i in range(len(generations)):
            tmp_gen = generations[i]
            response1_score = re.search(response1_score_pattern, tmp_gen, re.DOTALL)
            if response1_score:
                try:
                    response1_score = int(response1_score.group(1))
                except Exception:
                    response1_score = None
            else:
                response1_score = None


            response2_score = re.search(response2_score_pattern, tmp_gen, re.DOTALL)
            if response2_score:
                try:
                    response2_score = int(response2_score.group(1))
                except Exception:
                    response2_score = None
            else:
                response2_score = None

            ratio = re.search(ratio_pattern, tmp_gen)
            if ratio:
                try:
                    ratio = float(ratio.group(1))
                except Exception:
                    ratio = None
            else:
                ratio = None
        

            #quality_score > 1 means better current with higher quality
            if response1_score is not None and response2_score is not None:
                if response2_score == 0:
                    if ratio is not None:
                        quality_score = ratio
                    else:
                        quality_score = 60
                else:
                    quality_score = response1_score / response2_score
            elif ratio is not None:
                quality_score = ratio
            else:
                print("Warning: can not get the quality score!!!")
                quality_score = 1.0
        

            #length_num > 1 means better current with lower error
            if length_score(pred_words2[i], words[i]) == 0 and length_score(pred_words1[i], words[i]) == 0:
                length_num = 1.0
            elif length_score(pred_words2[i], words[i]) == 0:
                length_num = 0.1
            elif length_score(pred_words1[i], words[i]) == 0:
                length_num = 10
            else:
                length_num = length_score(pred_words2[i], words[i]) / length_score(pred_words1[i], words[i])

            final_score = length_num * (quality_score ** importance_factor)
            final_scores.append(final_score)



            if response1_score is not None:
                ind_q_score = response1_score / 60
            else:
                ind_q_score = 0.8
            independ_scores.append(ind_q_score)


    elif task == 'mtbench':
        assert instruction[0] is not None
        assert category[0] is not None

        tokens = []
        for i in range(len(instruction)):
            message = []

            if category[i] in ['reasoning', 'math', 'coding']:
                system_prompt = "You are a highly efficient assistant, who evaluates and selects the best large language model (LLMs) based on the quality of their responses to a given instruction. "\
                                "This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers."
                user_message = "I require a leaderboard for various large language models. "\
                    "I'll provide you with an instruction given to these models and their corresponding responses. "\
                    "Your task is to assess these responses, provide a quantitative assessment of their performance based on the criteria below, and select the model that produces the best output from a human perspective. "\
                    "Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.\n\n"\
                    f"Instruction:\n{instruction[i]}\n\nResponse 1:\n{prediction1[i]}\n\nResponse 2:\n{prediction2[i]}\n\n"\
                    "Evaluation Criteria (each scored on a scale of 1-10, with 10 being the best):\n"\
                    "1. Correctness: Is the answer logically sound, factually accurate, and free from errors?\n"\
                    "2. Helpfulness: Does the response directly address the instruction and provide meaningful assistance?\n"\
                    "3. Clarity: Is the response well-structured and easy to understand?\n"\
                    "4. Efficiency: Does the response provide an optimal solution without unnecessary complexity?\n"\
                    "5. Completeness: Does the response fully cover the instruction's requirements and edge cases?\n"\
                    "6. Robustness: Can the response handle ambiguity or complexity in the instruction?\n\n"\
                    "Tasks:\n"\
                    "* Score each response based on the above criteria.\n"\
                    "* Calculate an overall score for each response as the sum of all criteria scores (maximum 60).\n"\
                    "* Conclude by identifying which response is better overall.\n"\
                    "* Calculate a score ratio of Response 1 to Response 2 (Response 1 Score ÷ Response 2 Score).\n\n"\
                    "Output Format:\n"\
                    "#### Response 1:\n"\
                    "1. Correctness: [Score]/10\n"\
                    "2. Helpfulness: [Score]/10\n"\
                    "3. Clarity: [Score]/10\n"\
                    "4. Efficiency: [Score]/10\n"\
                    "5. Completeness: [Score]/10\n"\
                    "6. Robustness: [Score]/10\n"\
                    "**Overall Score:** [Total Score]/60\n\n"\
                    "#### Response 2:\n"\
                    "1. Correctness: [Score]/10\n"\
                    "2. Helpfulness: [Score]/10\n"\
                    "3. Clarity: [Score]/10\n"\
                    "4. Efficiency: [Score]/10\n"\
                    "5. Completeness: [Score]/10\n"\
                    "6. Robustness: [Score]/10\n"\
                    "**Overall Score:** [Total Score]/60\n\n"\
                    "### Conclusion:\n"\
                    "- **Better Response:** [Response 1/Response 2].\n"\
                    "- **Score Ratio (Response 1 ÷ Response 2):** [Ratio, rounded to two decimal places].\n"
            else:
                system_prompt = "You are a highly efficient assistant, who evaluates and selects the best large language model (LLMs) based on the quality of their responses to a given instruction. "\
                                "This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers."
                user_message = "I require a leaderboard for various large language models. "\
                    "I'll provide you with an instruction given to these models and their corresponding responses. "\
                    "Your task is to assess these responses, provide a quantitative assessment of their performance based on the criteria below, and select the model that produces the best output from a human perspective. "\
                    "Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.\n\n"\
                    f"Instruction:\n{instruction[i]}\n\nResponse 1:\n{prediction1[i]}\n\nResponse 2:\n{prediction2[i]}\n\n"\
                    "Evaluation Criteria (each scored on a scale of 1-10, with 10 being the best):\n"\
                    "1. Helpfulness: Does the response directly address the instruction and provide meaningful assistance?\n"\
                    "2. Relevance: Does the response stay on topic and avoid unnecessary or unrelated information?\n"\
                    "3. Accuracy: Is the information in the response factually correct and free of errors?\n"\
                    "4. Depth: Does the response demonstrate a deep understanding of the topic, including nuanced explanations where relevant?\n"\
                    "5. Creativity: Does the response display originality, creativity, or a unique approach to addressing the instruction?\n"\
                    "6. Level of Detail: Is the response sufficiently detailed, providing comprehensive and thorough explanations where necessary?\n\n"\
                    "Tasks:\n"\
                    "* Score each response based on the above criteria.\n"\
                    "* Calculate an overall score for each response as the sum of all criteria scores (maximum 60).\n"\
                    "* Conclude by identifying which response is better overall.\n"\
                    "* Calculate a score ratio of Response 1 to Response 2 (Response 1 Score ÷ Response 2 Score).\n\n"\
                    "Output Format:\n"\
                    "#### Response 1:\n"\
                    "1. Helpfulness: [Score]/10\n"\
                    "2. Relevance: [Score]/10\n"\
                    "3. Accuracy: [Score]/10\n"\
                    "4. Depth: [Score]/10\n"\
                    "5. Creativity: [Score]/10\n"\
                    "6. Level of Detail: [Score]/10\n"\
                    "**Overall Score:** [Total Score]/60\n\n"\
                    "#### Response 2:\n"\
                    "1. Helpfulness: [Score]/10\n"\
                    "2. Relevance: [Score]/10\n"\
                    "3. Accuracy: [Score]/10\n"\
                    "4. Depth: [Score]/10\n"\
                    "5. Creativity: [Score]/10\n"\
                    "6. Level of Detail: [Score]/10\n"\
                    "**Overall Score:** [Total Score]/60\n\n"\
                    "### Conclusion:\n"\
                    "- **Better Response:** [Response 1/Response 2].\n"\
                    "- **Score Ratio (Response 1 ÷ Response 2):** [Ratio, rounded to two decimal places].\n"
            if system_prompt is not None:
                message.append({"role": "system", "content":system_prompt})
            message.append({"role":"user", "content":user_message})
            token = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            tokens.append(token)
        
        tokenization = tokenizer(tokens, padding=True, padding_side='left', add_special_tokens=False, return_tensors='pt')
        input_ids, attention_mask = tokenization.input_ids, tokenization.attention_mask
        batch_size, input_length = input_ids.shape

        output = model.generate(input_ids.to(model.device), attention_mask=attention_mask.to(model.device), max_new_tokens=max_new_tokens, use_cache=True).to('cpu')

        generations = [tokenizer.decode(output[b, input_length:], skip_special_tokens=True) for b in range(batch_size)]

        response1_score_pattern = r"Response 1:\s.*?Overall Score:\**\s(\d+)/60"
        response2_score_pattern = r"Response 2:\s.*?Overall Score:\**\s(\d+)/60"
        ratio_pattern = r"Score Ratio \(Response 1 ÷ Response 2\):\**\s(\d+.\d+)"

        final_scores = []
        independ_scores = []
        for i in range(len(generations)):
            tmp_gen = generations[i]
            response1_score = re.search(response1_score_pattern, tmp_gen, re.DOTALL)
            if response1_score:
                try:
                    response1_score = int(response1_score.group(1))
                except Exception:
                    response1_score = None
            else:
                response1_score = None


            response2_score = re.search(response2_score_pattern, tmp_gen, re.DOTALL)
            if response2_score:
                try:
                    response2_score = int(response2_score.group(1))
                except Exception:
                    response2_score = None
            else:
                response2_score = None

            ratio = re.search(ratio_pattern, tmp_gen)
            if ratio:
                try:
                    ratio = float(ratio.group(1))
                except Exception:
                    ratio = None
            else:
                ratio = None
        

            #quality_score > 1 means better current with higher quality
            if response1_score is not None and response2_score is not None:
                if response2_score == 0:
                    if ratio is not None:
                        quality_score = ratio
                    else:
                        quality_score = 60
                else:
                    quality_score = response1_score / response2_score
            elif ratio is not None:
                quality_score = ratio
            else:
                print("Warning: can not get the quality score!!!")
                quality_score = 1.0
        

            #length_num > 1 means better current with lower error
            if length_score(pred_words2[i], words[i]) == 0 and length_score(pred_words1[i], words[i]) == 0:
                length_num = 1.0
            elif length_score(pred_words2[i], words[i]) == 0:
                length_num = 0.1
            elif length_score(pred_words1[i], words[i]) == 0:
                length_num = 10
            else:
                length_num = length_score(pred_words2[i], words[i]) / length_score(pred_words1[i], words[i])

            final_score = length_num * (quality_score ** importance_factor)
            final_scores.append(final_score)



            if response1_score is not None:
                ind_q_score = response1_score / 60
            else:
                ind_q_score = 0.8
            independ_scores.append(ind_q_score)
    else:
        raise Exception

        
    return final_scores, independ_scores





def process_score_api(input_dict, client, model, length_score, task='summary', importance_factor=1.0):
    #1 current 
    #2 previous
    if isinstance(input_dict, dict):
        article = input_dict['article']
        prediction1, prediction2 = input_dict['prediction_pair']
        words = input_dict['words']
        pred_words1, pred_words2 = input_dict['pred_words_pair']
        instruction = input_dict['instruction']
        category = input_dict['category']

    
    if task == 'summary':
        assert article is not None

        message = []
        system_prompt = "You are a powerful evaluator for abstractive summarization."
        if system_prompt is not None:
            message.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        user_message = "I need to compare and evaluate the quality of two summaries generated for a given document. "\
            "Please provide a quantitative assessment of their performance based on the criteria below.\n\n"\
            f"Document:\n{article}\n\nSummary 1:\n{prediction1}\n\nSummary 2:\n{prediction2}\n\n"\
            "Evaluation Criteria (each scored on a scale of 1-10, with 10 being the best):\n"\
            "1. Information Coverage: Does the summary include the most important and critical information from the document?\n"\
            "2. Linguistic Fluency: Are the sentences in the summary fluent, natural, and grammatically correct?\n"\
            "3. Conciseness: Does the summary avoid redundancy while retaining key information?\n"\
            "4. Logical Coherence: Is the summary well-structured with clear and logical flow?\n"\
            "5. Faithfulness: Does the summary accurately reflect the facts in the original document without adding false or misleading information?\n\n"\
            "Instructions:\n"\
            "* Score each summary based on the above criteria.\n"\
            "* Calculate an overall score for each summary as the sum of all criteria scores (maximum 50).\n"\
            "* Conclude by identifying which summary is better overall.\n"\
            "* Calculate a score ratio of Summary 1 to Summary 2 (Summary 1 Score ÷ Summary 2 Score).\n\n"\
            "Output Format:\n"\
            "#### Summary 1:\n"\
            "1. Information Coverage: [Score]/10\n"\
            "2. Linguistic Fluency: [Score]/10\n"\
            "3. Conciseness: [Score]/10\n"\
            "4. Logical Coherence: [Score]/10\n"\
            "5. Faithfulness: [Score]/10\n"\
            "**Overall Score:** [Total Score]/50\n\n"\
            "#### Summary 2:\n"\
            "1. Information Coverage: [Score]/10\n"\
            "2. Linguistic Fluency: [Score]/10\n"\
            "3. Conciseness: [Score]/10\n"\
            "4. Logical Coherence: [Score]/10\n"\
            "5. Faithfulness: [Score]/10\n"\
            "**Overall Score:** [Total Score]/50\n\n"\
            "### Conclusion:\n"\
            "- **Better Summary:** [Summary 1/Summary 2].\n"\
            "- **Score Ratio (Summary 1 ÷ Summary 2):** [Ratio, rounded to two decimal places].\n"
        
        message.append({"role":"user", "content": [{"type": "text", "text": user_message}]})
        response = client.chat.completions.create(
            model=model,
            messages=message,
        )
        generation = response.choices[0].message.content



        summary1_score_pattern = r"Summary 1:\s.*?Overall Score:\**\s(\d+)/50"
        summary2_score_pattern = r"Summary 2:\s.*?Overall Score:\**\s(\d+)/50"
        ratio_pattern = r"Score Ratio \(Summary 1 ÷ Summary 2\):\**\s(\d+.\d+)"

        summary1_score = re.search(summary1_score_pattern, generation, re.DOTALL)
        if summary1_score:
            try:
                summary1_score = int(summary1_score.group(1))
            except Exception:
                summary1_score = None
        else:
            summary1_score = None


        summary2_score = re.search(summary2_score_pattern, generation, re.DOTALL)
        if summary2_score:
            try:
                summary2_score = int(summary2_score.group(1))
            except Exception:
                summary2_score = None
        else:
            summary2_score = None

        ratio = re.search(ratio_pattern, generation)
        if ratio:
            try:
                ratio = float(ratio.group(1))
            except Exception:
                ratio = None
        else:
            ratio = None
        

        #quality_score > 1 means better current with higher quality
        if summary1_score is not None and summary2_score is not None:
            if summary2_score == 0:
                if ratio is not None:
                    quality_score = ratio
                else:
                    quality_score = 50
            else:
                quality_score = summary1_score / summary2_score
        elif ratio is not None:
            quality_score = ratio
        else:
            print("Warning: can not get the quality score!!!")
            quality_score = 1.0
        

        #length_num > 1 means better current with lower error
        if length_score(pred_words2, words) == 0 and length_score(pred_words1, words) == 0:
            length_num = 1.0
        elif length_score(pred_words2, words) == 0:
            length_num = 0.1
        elif length_score(pred_words1, words) == 0:
            length_num = 10
        else:
            length_num = length_score(pred_words2, words) / length_score(pred_words1, words)

        final_score = length_num * (quality_score ** importance_factor)


        if summary1_score is not None:
            ind_q_score1 = summary1_score / 50
        else:
            ind_q_score1 = 0.8

        if summary2_score is not None:
            ind_q_score2 = summary2_score / 50
        else:
            ind_q_score2 = 0.8


    elif task == 'alpaca':
        assert instruction is not None

        message = []
        system_prompt = "You are a highly efficient assistant, who evaluates and selects the best large language model (LLMs) based on the quality of their responses to a given instruction. "\
                        "This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers."
        if system_prompt is not None:
            message.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        user_message = "I require a leaderboard for various large language models. "\
            "I'll provide you with an instruction given to these models and their corresponding responses. "\
            "Your task is to assess these responses, provide a quantitative assessment of their performance based on the criteria below, and select the model that produces the best output from a human perspective. "\
            "Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.\n\n"\
            f"Instruction:\n{instruction}\n\nResponse 1:\n{prediction1}\n\nResponse 2:\n{prediction2}\n\n"\
            "Evaluation Criteria (each scored on a scale of 1-10, with 10 being the best):\n"\
            "1. Helpfulness: Does the response directly address the instruction and provide meaningful assistance?\n"\
            "2. Relevance: Does the response stay on topic and avoid unnecessary or unrelated information?\n"\
            "3. Accuracy: Is the information in the response factually correct and free of errors?\n"\
            "4. Depth: Does the response demonstrate a deep understanding of the topic, including nuanced explanations where relevant?\n"\
            "5. Creativity: Does the response display originality, creativity, or a unique approach to addressing the instruction?\n"\
            "6. Level of Detail: Is the response sufficiently detailed, providing comprehensive and thorough explanations where necessary?\n\n"\
            "Tasks:\n"\
            "* Score each response based on the above criteria.\n"\
            "* Calculate an overall score for each response as the sum of all criteria scores (maximum 60).\n"\
            "* Conclude by identifying which response is better overall.\n"\
            "* Calculate a score ratio of Response 1 to Response 2 (Response 1 Score ÷ Response 2 Score).\n\n"\
            "Output Format:\n"\
            "#### Response 1:\n"\
            "1. Helpfulness: [Score]/10\n"\
            "2. Relevance: [Score]/10\n"\
            "3. Accuracy: [Score]/10\n"\
            "4. Depth: [Score]/10\n"\
            "5. Creativity: [Score]/10\n"\
            "6. Level of Detail: [Score]/10\n"\
            "**Overall Score:** [Total Score]/60\n\n"\
            "#### Response 2:\n"\
            "1. Helpfulness: [Score]/10\n"\
            "2. Relevance: [Score]/10\n"\
            "3. Accuracy: [Score]/10\n"\
            "4. Depth: [Score]/10\n"\
            "5. Creativity: [Score]/10\n"\
            "6. Level of Detail: [Score]/10\n"\
            "**Overall Score:** [Total Score]/60\n\n"\
            "### Conclusion:\n"\
            "- **Better Response:** [Response 1/Response 2].\n"\
            "- **Score Ratio (Response 1 ÷ Response 2):** [Ratio, rounded to two decimal places].\n"
    
        message.append({"role":"user", "content": [{"type": "text", "text": user_message}]})
        response = client.chat.completions.create(
            model=model,
            messages=message,
        )
        generation = response.choices[0].message.content
        

        response1_score_pattern = r"Response 1:\s.*?Overall Score:\**\s(\d+)/60"
        response2_score_pattern = r"Response 2:\s.*?Overall Score:\**\s(\d+)/60"
        ratio_pattern = r"Score Ratio \(Response 1 ÷ Response 2\):\**\s(\d+.\d+)"


        response1_score = re.search(response1_score_pattern, generation, re.DOTALL)
        if response1_score:
            try:
                response1_score = int(response1_score.group(1))
            except Exception:
                response1_score = None
        else:
            response1_score = None


        response2_score = re.search(response2_score_pattern, generation, re.DOTALL)
        if response2_score:
            try:
                response2_score = int(response2_score.group(1))
            except Exception:
                response2_score = None
        else:
            response2_score = None

        ratio = re.search(ratio_pattern, generation)
        if ratio:
            try:
                ratio = float(ratio.group(1))
            except Exception:
                ratio = None
        else:
            ratio = None
        

        #quality_score > 1 means better current with higher quality
        if response1_score is not None and response2_score is not None:
            if response2_score == 0:
                if ratio is not None:
                    quality_score = ratio
                else:
                    quality_score = 60
            else:
                quality_score = response1_score / response2_score
        elif ratio is not None:
            quality_score = ratio
        else:
            print("Warning: can not get the quality score!!!")
            quality_score = 1.0
        

        #length_num > 1 means better current with lower error
        if length_score(pred_words2, words) == 0 and length_score(pred_words1, words) == 0:
            length_num = 1.0
        elif length_score(pred_words2, words) == 0:
            length_num = 0.1
        elif length_score(pred_words1, words) == 0:
            length_num = 10
        else:
            length_num = length_score(pred_words2, words) / length_score(pred_words1, words)

        final_score = length_num * (quality_score ** importance_factor)



        if response1_score is not None:
            ind_q_score1 = response1_score / 60
        else:
            ind_q_score1 = 0.8

        if response2_score is not None:
            ind_q_score2 = response2_score / 60
        else:
            ind_q_score2 = 0.8



    elif task == 'mtbench':
        assert instruction is not None
        assert category is not None

        message = []

        if category in ['reasoning', 'math', 'coding']:
            system_prompt = "You are a highly efficient assistant, who evaluates and selects the best large language model (LLMs) based on the quality of their responses to a given instruction. "\
                            "This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers."
            user_message = "I require a leaderboard for various large language models. "\
                "I'll provide you with an instruction given to these models and their corresponding responses. "\
                "Your task is to assess these responses, provide a quantitative assessment of their performance based on the criteria below, and select the model that produces the best output from a human perspective. "\
                "Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.\n\n"\
                f"Instruction:\n{instruction}\n\nResponse 1:\n{prediction1}\n\nResponse 2:\n{prediction2}\n\n"\
                "Evaluation Criteria (each scored on a scale of 1-10, with 10 being the best):\n"\
                "1. Correctness: Is the answer logically sound, factually accurate, and free from errors?\n"\
                "2. Helpfulness: Does the response directly address the instruction and provide meaningful assistance?\n"\
                "3. Clarity: Is the response well-structured and easy to understand?\n"\
                "4. Efficiency: Does the response provide an optimal solution without unnecessary complexity?\n"\
                "5. Completeness: Does the response fully cover the instruction's requirements and edge cases?\n"\
                "6. Robustness: Can the response handle ambiguity or complexity in the instruction?\n\n"\
                "Tasks:\n"\
                "* Score each response based on the above criteria.\n"\
                "* Calculate an overall score for each response as the sum of all criteria scores (maximum 60).\n"\
                "* Conclude by identifying which response is better overall.\n"\
                "* Calculate a score ratio of Response 1 to Response 2 (Response 1 Score ÷ Response 2 Score).\n\n"\
                "Output Format:\n"\
                "#### Response 1:\n"\
                "1. Correctness: [Score]/10\n"\
                "2. Helpfulness: [Score]/10\n"\
                "3. Clarity: [Score]/10\n"\
                "4. Efficiency: [Score]/10\n"\
                "5. Completeness: [Score]/10\n"\
                "6. Robustness: [Score]/10\n"\
                "**Overall Score:** [Total Score]/60\n\n"\
                "#### Response 2:\n"\
                "1. Correctness: [Score]/10\n"\
                "2. Helpfulness: [Score]/10\n"\
                "3. Clarity: [Score]/10\n"\
                "4. Efficiency: [Score]/10\n"\
                "5. Completeness: [Score]/10\n"\
                "6. Robustness: [Score]/10\n"\
                "**Overall Score:** [Total Score]/60\n\n"\
                "### Conclusion:\n"\
                "- **Better Response:** [Response 1/Response 2].\n"\
                "- **Score Ratio (Response 1 ÷ Response 2):** [Ratio, rounded to two decimal places].\n"
        else:
            system_prompt = "You are a highly efficient assistant, who evaluates and selects the best large language model (LLMs) based on the quality of their responses to a given instruction. "\
                            "This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers."
            user_message = "I require a leaderboard for various large language models. "\
                "I'll provide you with an instruction given to these models and their corresponding responses. "\
                "Your task is to assess these responses, provide a quantitative assessment of their performance based on the criteria below, and select the model that produces the best output from a human perspective. "\
                "Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.\n\n"\
                f"Instruction:\n{instruction}\n\nResponse 1:\n{prediction1}\n\nResponse 2:\n{prediction2}\n\n"\
                "Evaluation Criteria (each scored on a scale of 1-10, with 10 being the best):\n"\
                "1. Helpfulness: Does the response directly address the instruction and provide meaningful assistance?\n"\
                "2. Relevance: Does the response stay on topic and avoid unnecessary or unrelated information?\n"\
                "3. Accuracy: Is the information in the response factually correct and free of errors?\n"\
                "4. Depth: Does the response demonstrate a deep understanding of the topic, including nuanced explanations where relevant?\n"\
                "5. Creativity: Does the response display originality, creativity, or a unique approach to addressing the instruction?\n"\
                "6. Level of Detail: Is the response sufficiently detailed, providing comprehensive and thorough explanations where necessary?\n\n"\
                "Tasks:\n"\
                "* Score each response based on the above criteria.\n"\
                "* Calculate an overall score for each response as the sum of all criteria scores (maximum 60).\n"\
                "* Conclude by identifying which response is better overall.\n"\
                "* Calculate a score ratio of Response 1 to Response 2 (Response 1 Score ÷ Response 2 Score).\n\n"\
                "Output Format:\n"\
                "#### Response 1:\n"\
                "1. Helpfulness: [Score]/10\n"\
                "2. Relevance: [Score]/10\n"\
                "3. Accuracy: [Score]/10\n"\
                "4. Depth: [Score]/10\n"\
                "5. Creativity: [Score]/10\n"\
                "6. Level of Detail: [Score]/10\n"\
                "**Overall Score:** [Total Score]/60\n\n"\
                "#### Response 2:\n"\
                "1. Helpfulness: [Score]/10\n"\
                "2. Relevance: [Score]/10\n"\
                "3. Accuracy: [Score]/10\n"\
                "4. Depth: [Score]/10\n"\
                "5. Creativity: [Score]/10\n"\
                "6. Level of Detail: [Score]/10\n"\
                "**Overall Score:** [Total Score]/60\n\n"\
                "### Conclusion:\n"\
                "- **Better Response:** [Response 1/Response 2].\n"\
                "- **Score Ratio (Response 1 ÷ Response 2):** [Ratio, rounded to two decimal places].\n"
        if system_prompt is not None:
            message.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        message.append({"role":"user", "content": [{"type": "text", "text": user_message}]})
        response = client.chat.completions.create(
            model=model,
            messages=message,
        )
        generation = response.choices[0].message.content
        

        response1_score_pattern = r"Response 1:\s.*?Overall Score:\**\s(\d+)/60"
        response2_score_pattern = r"Response 2:\s.*?Overall Score:\**\s(\d+)/60"
        ratio_pattern = r"Score Ratio \(Response 1 ÷ Response 2\):\**\s(\d+.\d+)"



        response1_score = re.search(response1_score_pattern, generation, re.DOTALL)
        if response1_score:
            try:
                response1_score = int(response1_score.group(1))
            except Exception:
                response1_score = None
        else:
            response1_score = None


        response2_score = re.search(response2_score_pattern, generation, re.DOTALL)
        if response2_score:
            try:
                response2_score = int(response2_score.group(1))
            except Exception:
                response2_score = None
        else:
            response2_score = None

        ratio = re.search(ratio_pattern, generation)
        if ratio:
            try:
                ratio = float(ratio.group(1))
            except Exception:
                ratio = None
        else:
            ratio = None
    

        #quality_score > 1 means better current with higher quality
        if response1_score is not None and response2_score is not None:
            if response2_score == 0:
                if ratio is not None:
                    quality_score = ratio
                else:
                    quality_score = 60
            else:
                quality_score = response1_score / response2_score
        elif ratio is not None:
            quality_score = ratio
        else:
            print("Warning: can not get the quality score!!!")
            quality_score = 1.0
        

        #length_num > 1 means better current with lower error
        if length_score(pred_words2, words) == 0 and length_score(pred_words1, words) == 0:
            length_num = 1.0
        elif length_score(pred_words2, words) == 0:
            length_num = 0.1
        elif length_score(pred_words1, words) == 0:
            length_num = 10
        else:
            length_num = length_score(pred_words2, words) / length_score(pred_words1, words)

        final_score = length_num * (quality_score ** importance_factor)



        if response1_score is not None:
            ind_q_score1 = response1_score / 60
        else:
            ind_q_score1 = 0.8

        if response2_score is not None:
            ind_q_score2 = response2_score / 60
        else:
            ind_q_score2 = 0.8

    
    else:
        raise Exception
    
    return final_score, (ind_q_score1, ind_q_score2)