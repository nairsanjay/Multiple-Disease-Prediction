from flask import Flask, render_template, request, flash, redirect
import pickle
import numpy as np

app = Flask(__name__)

def pre(values, dic):
    values = np.asarray(values)
    if len(values) == 8:
        return pickle.load(open('models/diabetes.pkl', 'rb')).predict(values.reshape(1, -1))[0]
    elif len(values) == 26:
        return pickle.load(open('models/cancer.pkl', 'rb')).predict(values.reshape(1, -1))[0]
    elif len(values) == 13:
        return pickle.load(open('models/heart.pkl', 'rb')).predict(values.reshape(1, -1))[0]
    elif len(values) == 18:
        return pickle.load(open('models/kidney.pkl', 'rb')).predict(values.reshape(1, -1))[0]
    elif len(values) == 10:
        return pickle.load(open('models/liver.pkl', 'rb')).predict(values.reshape(1, -1))[0]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            pred = pre(list(map(float, list(request.form.to_dict().values()))), request.form.to_dict())
    except:
        return render_template("home.html", message="Please enter valid Data")

    return render_template('predict.html', pred = pred)


@app.route("/pneumoniapredict", methods = ['POST', 'GET'])
def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            img = np.asarray(Image.open(request.files['image']).convert('L').resize((36, 36))).reshape(
                (1, 36, 36, 1)) / 255.0
            pred2 = np.argmax(load_model("models/pneumonia.h5").predict(img)[0])
        except:
            message = "Please upload an Image"
            return render_template('pneumonia.html', message = message)
    return render_template('pneumonia_predict.html', pred = pred2)


if __name__ == '__main__':
	app.run(debug = True)