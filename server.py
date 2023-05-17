from flask import Flask, render_template, request
from ai_models import AI_models
from deep_translator import GoogleTranslator

AI_models = AI_models()
app = Flask(__name__)

@app.route("/")
@app.route("/index")
@app.route("/home")
def home():
    return render_template('index.html')

@app.route("/linguoBridge",methods=["GET"])
def linguoBridge():
    if request.method == "GET":
        query = request.args.get('query')
        if query == None:
            return render_template('translate.html')
        else:
            print(query.replace("%0D%0A", "\n"))
            translated = GoogleTranslator(source='auto', target='hi').translate(query.replace("%0D%0A", "\n"))
        return render_template('translate.html',query=query,translated=translated)

@app.route("/paraSimplify")
def paraSimplify():
    if request.method == "GET":
        query = request.args.get('query')
        if query == None:
            return render_template('rephrase.html')
        else:
            print(query.replace("%0D%0A", "\n"))
            paraphrases = AI_models.paraphrase(query,max_length=len(query))
        return render_template('rephrase.html',query=query,paraphrases=paraphrases)
    
@app.route("/summary")
def summary():
    if request.method == "GET":
        query = request.args.get('query')
        if query == None:
            return render_template('summarize.html')
        else:
            print(query.replace("%0D%0A", "\n"))
            summary_list=AI_models.summarizer(query,max_length=len(query))
        return render_template('summarize.html',query=query,summary_list=summary_list)
    
@app.route("/queryMaster")
def queryMaster():
    if request.method == "GET":
        question = request.args.get('question')
        context = request.args.get('context')
        if question == None and context==None:
            return render_template('question_answer.html')
        else:
            print(question)
            print(context)
            answer = AI_models.question_answering(question,context)
            return render_template('question_answer.html',question=question,context=context,answer=answer)
    


    
@app.route("/ExamInsider",methods=["GET"])
def ExamInsider():
    if request.method == "GET":
        subject = request.args.get('subject')
        if subject == None:
            return render_template('question_analyzer.html')
        else:
            print(subject)
            data = AI_models.question_analyzer(subject)
            print(data)
            return render_template('question_analyzer.html',subject=subject,data=data)


if __name__=="__main__":
    app.run(debug=True)