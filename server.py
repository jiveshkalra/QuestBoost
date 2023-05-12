from flask import Flask, render_template, request
from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,pipeline, AutoModelForQuestionAnswering
import openai
import json

device = "cpu"
openai.api_key = "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"


app = Flask(__name__)

def question_answer_model(question,context):
    qa_model_name = "deepset/roberta-base-squad2"
    # a) Load model & tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    QA_input = {'question':question,'context':context}
    return nlp(QA_input)['answer']
    

""" Open AI 
def question_answer_model(question,context):
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"Question: {question}\nContext: {context}\nAnswer:",
        temperature=0.5,
        max_tokens=100,
        n=1,
        stop=None,
        timeout=10
    )

    # Extract the answer from the API response
    answer = response.choices[0].text.strip()
    return answer
"""
def paraphrase(
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


@app.route("/")
@app.route("/index")
@app.route("/home")
def home():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/login")
def login():
    return render_template('login.html')

@app.route("/question_answer")
def question_answer():
    if request.method == "GET":
        question = request.args.get('question')
        context = request.args.get('context')
        if question == None and context==None:
            return render_template('question_answer.html')
        else:
            print(question)
            print(context)
            answer = question_answer_model(question,context)
            return render_template('question_answer.html',question=question,context=context,answer=answer)

    
@app.route("/contact")
def contact():
    return render_template('contact.html')

@app.route("/rephrase")
def rephrase():
    if request.method == "GET":
        query = request.args.get('query')
        if query == None:
            return render_template('rephrase.html')
        else:
            print(query.replace("%0D%0A", "\n"))
            paraphrases = paraphrase(query,max_length=len(query))
        return render_template('rephrase.html',query=query,paraphrases=paraphrases)

@app.route("/translate",methods=["GET"])
def translate():
    if request.method == "GET":
        query = request.args.get('query')
        if query == None:
            return render_template('translate.html')
        else:
            print(query.replace("%0D%0A", "\n"))
            translated = GoogleTranslator(source='auto', target='hi').translate(query.replace("%0D%0A", "\n"))
        return render_template('translate.html',query=query,translated=translated)
        

if __name__=="__main__":
    app.run(debug=True)