
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,pipeline, AutoModelForQuestionAnswering

def summarizer_model(query,max_length=1000):
    tokenizer = AutoTokenizer.from_pretrained("pszemraj/pegasus-x-large-book-summary")
    model_1 = AutoModelForSeq2SeqLM.from_pretrained("pszemraj/pegasus-x-large-book-summary")
    input_ids = tokenizer.encode(query, truncation=True, padding=True, return_tensors="pt")
    summary_ids = model_1.generate(input_ids, max_length=max_length, num_beams=4, early_stopping=True)
    summary_1 = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    model_2 = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
    summary_2 = model_2(query)[0]['summary_text']
    return [summary_1,summary_2]


