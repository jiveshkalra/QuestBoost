from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,pipeline, AutoModelForQuestionAnswering
import os
import json

device = "cpu"

class AI_models():
    def summarizer(self,query,max_length=1000):
        tokenizer = AutoTokenizer.from_pretrained("pszemraj/pegasus-x-large-book-summary")
        model_1 = AutoModelForSeq2SeqLM.from_pretrained("pszemraj/pegasus-x-large-book-summary")
        input_ids = tokenizer.encode(query, truncation=True, padding=True, return_tensors="pt")
        summary_ids = model_1.generate(input_ids, max_length=max_length, num_beams=4, early_stopping=True)
        summary_1 = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        model_2 = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
        summary_2 = model_2(query)[0]['summary_text']
        return [summary_1,summary_2]
    
    def question_analyzer(self,subject):
        data_files = ['eng']
        if subject in data_files:
            json_path = os.path.join('static', 'json', f'{subject}.json')
            with open(json_path) as file:
                file_contents = file.read()
                parsed_json = json.loads(file_contents)
                return parsed_json
        else:
            return None

    def question_answering(self,question,context):
        qa_model_name = "deepset/roberta-base-squad2"
        # a) Load model & tokenizer
        model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
        tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
        nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
        QA_input = {'question':question,'context':context}
        return nlp(QA_input)['answer']
    
    def paraphrase(
        self,
        question,
        num_beams=5,
        num_beam_groups=5,
        num_return_sequences=5,
        repetition_penalty=10.0,
        diversity_penalty=3.0,
        no_repeat_ngram_size=2,
        temperature=0.7,
        max_length=1000
    ):
        paraphrase_tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
        paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)
        input_ids = paraphrase_tokenizer(
            f'paraphrase: {question}',
            return_tensors="pt", padding="longest",
            max_length=max_length,
            truncation=True,
        ).input_ids
            
        outputs = paraphrase_model.generate(
            input_ids.to('cpu'), temperature=temperature, repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=num_beams, num_beam_groups=num_beam_groups,
            max_length=max_length, diversity_penalty=diversity_penalty
        )

        res = paraphrase_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return res