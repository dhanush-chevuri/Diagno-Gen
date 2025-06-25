
import joblib
from flask import Flask, render_template, redirect, url_for, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import pandas as pd
import pickle
from markdown import markdown
import requests
import pytesseract
from PIL import Image
import tempfile
from flask import jsonify
from pprint import pprint
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
from werkzeug.utils import secure_filename
import pdfplumber
import io
import csv


import openpyxl
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from keras.models import load_model
model2 = load_model('model.h5')

import json
import random
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
prompt = """act like medical expert force yourself for classifying into a disease until medical report says person is health and tell the precautions to be taken
-give response in reporting tone and give any extra content just give
-dont give contents from report
-include only patient name and his indicators of diagnosis while explaining
-disease diagnosis and reason-precautions
"""
IMAGE_FORMATS = ['jpeg', 'jpg', 'png', 'bmp', 'tiff', 'tif']
# Get API key from environment variable or set directly (not recommended for production)
api_key = "AIzaSyA40CLZfuiuVjGhoHxMu79Pw5LGSw_3M30"

# Add API key as query parameter
url = f"{url}?key={api_key}"

###############################################################################


filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
model1 = pickle.load(open('model1.pkl', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

app = Flask(__name__,template_folder=".")
app.config['SECRET_KEY'] = 'secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('remember me')


class RegisterForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/help')
def help():
    return render_template("help.html")


@app.route('/terms')
def terms():
    return render_template("tc.html")


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('dashboard'))

        return render_template("login.html", form=form)
    return render_template("login.html", form=form)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return redirect("/login")
    return render_template('signup.html', form=form)


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


@app.route("/disindex")

def disindex():
    return render_template("disindex.html")

@app.route('/chatbot')
@login_required
def chatbot():
    

# Set up the API key


    print("Gemini Chatbot: Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break



    return render_template("chatbot.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

@app.route("/cancer")
@login_required
def cancer():
    return render_template("cancer.html")


@app.route("/diabetes")
@login_required
def diabetes():
    return render_template("diabetes.html")


@app.route("/heart")
@login_required
def heart():
    return render_template("heart.html")


@app.route("/kidney")
@login_required
def kidney():
    return render_template("kidney.html")


@app.route("/parkinsons")
@login_required
def parkinsons():
    return render_template("parkinsons.html")


@app.route('/predictparkinsons', methods=['POST'])
@login_required
def predictparkinsons():
    if request.method == "POST":
        # Get all the values from the form
        fo = float(request.form['fo'])
        fhi = float(request.form['fhi'])
        flo = float(request.form['flo'])
        jitter = float(request.form['jitter'])
        jitter_abs = float(request.form['jitter_abs'])
        rap = float(request.form['rap'])
        ppq = float(request.form['ppq'])
        jitter_ddp = float(request.form['jitter_ddp'])
        shimmer = float(request.form['shimmer'])
        shimmer_db = float(request.form['shimmer_db'])
        apq3 = float(request.form['apq3'])
        apq5 = float(request.form['apq5'])
        apq = float(request.form['apq'])
        shimmer_dda = float(request.form['shimmer_dda'])
        nhr = float(request.form['nhr'])
        hnr = float(request.form['hnr'])
        rpde = float(request.form['rpde'])
        dfa = float(request.form['dfa'])
        spread1 = float(request.form['spread1'])
        spread2 = float(request.form['spread2'])
        d2 = float(request.form['d2'])
        ppe = float(request.form['ppe'])

        # Make prediction
        prediction = parkinsons_model.predict([[fo, fhi, flo, jitter, jitter_abs, rap, ppq, jitter_ddp, 
                                             shimmer, shimmer_db, apq3, apq5, apq, shimmer_dda, nhr, hnr,
                                             rpde, dfa, spread1, spread2, d2, ppe]])

        if prediction[0] == 1:
            prediction_text = "Patient has a high risk of Parkinson's Disease, please consult your doctor immediately"
        else:
            prediction_text = "Patient has a low risk of Parkinson's Disease"

        return render_template("parkinsons_result.html", prediction_text=prediction_text)


def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if size == 7:
        loaded_model = joblib.load('kidney_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]


@app.route("/predictkidney",  methods=['GET', 'POST'])
def predictkidney():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if len(to_predict_list) == 7:
            result = ValuePredictor(to_predict_list, 7)
    if(int(result) == 1):
        prediction = "Patient has a high risk of Kidney Disease, please consult your doctor immediately"
    else:
        prediction = "Patient has a low risk of Kidney Disease"
    return render_template("kidney_result.html", prediction_text=prediction)


@app.route("/liver")
@login_required
def liver():
    return render_template("liver.html")


def ValuePred(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==7):
        result = model1.predict(to_predict)
    return result[0]


@app.route('/predictliver', methods=['POST'])
@login_required
def predictliver():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if len(to_predict_list) == 7:
            result = ValuePred(to_predict_list, 7)

    if int(result) == 1:
        prediction = "Patient has a high risk of Liver Disease, please consult your doctor immediately"
    else:
        prediction = "Patient has a low risk of Liver Disease"
    return render_template("liver_result.html", prediction_text=prediction)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/predict', methods=['POST'])
def predict():
    input_features = [int(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion',
                     'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
    if output == 4:
        res_val = "a high risk of Breast Cancer"
    else:
        res_val = "a low risk of Breast Cancer"

    return render_template('cancer_result.html', prediction_text='Patient has {}'.format(res_val))


##################################################################################

df1 = pd.read_csv('diabetes.csv')

# Renaming DiabetesPedigreeFunction as DPF
df1 = df1.rename(columns={'DiabetesPedigreeFunction': 'DPF'})

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
df_copy = df1.copy(deep=True)
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[['Glucose', 'BloodPressure',
                                                                                    'SkinThickness', 'Insulin',
                                                                                    'BMI']].replace(0, np.NaN)

# Replacing NaN value by mean, median depending upon distribution
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

# Model Building

X = df1.drop(columns='Outcome')
y = df1['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating Random Forest Model

classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

#####################################################################


@app.route('/predictt', methods=['POST'])
def predictt():
    if request.method == 'POST':
        preg = request.form['pregnancies']
        glucose = request.form['glucose']
        bp = request.form['bloodpressure']
        st = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)

        return render_template('diab_result.html', prediction=my_prediction)


############################################################################################################

@app.route('/predictheart', methods=['POST'])
def predictheart():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ["age", "trestbps", "chol", "thalach", "oldpeak", "sex_0",
                     "  sex_1", "cp_0", "cp_1", "cp_2", "cp_3", "  fbs_0",
                     "restecg_0", "restecg_1", "restecg_2", "exang_0", "exang_1",
                     "slope_0", "slope_1", "slope_2", "ca_0", "ca_1", "ca_2", "thal_1",
                     "thal_2", "thal_3"]

    df = pd.DataFrame(features_value, columns=features_name)
    output = model1.predict(df)

    if output == 1:
        res_val = "a high risk of Heart Disease"
    else:
        res_val = "a low risk of Heart Disease"

    return render_template('heart_result.html', prediction_text='Patient has {}'.format(res_val))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model2):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model2.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model2)
    res = getResponse(ints, intents)
    return res


@app.route('/report_analysis')
@login_required
def report_analysis():
    return render_template('report_analysis.html')

def extract_text_from_pdf(pdf_file):
    pdf_reader = pdfplumber.open(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_excel(excel_file):
    wb = openpyxl.load_workbook(excel_file)
    text = ""
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        for row in ws.rows:
            text += " ".join(str(cell.value) for cell in row if cell.value) + "\n"
    return text

def extract_text_from_csv(csv_file):
    text = ""
    csv_data = csv.reader(csv_file)
    for row in csv_data:
        text += " ".join(row) + "\n"
    return text

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return " ".join(tokens)

"""def analyze_text_for_diseases(text):
    results = {}
    
    # Keywords for different diseases
    disease_keywords = {
        'Diabetes': ['glucose', 'blood sugar', 'diabetes', 'insulin', 'hyperglycemia'],
        'Heart Disease': ['heart', 'cardiac', 'chest pain', 'angina', 'myocardial'],
        'Cancer': ['cancer', 'tumor', 'malignant', 'biopsy', 'oncology'],
        'Kidney Disease': ['kidney', 'renal', 'creatinine', 'glomerular', 'nephritis'],
        'Liver Disease': ['liver', 'hepatic', 'jaundice', 'cirrhosis', 'hepatitis']
    }
    
    # Analyze text for each disease
    for disease, keywords in disease_keywords.items():
        keyword_count = sum(1 for keyword in keywords if keyword in text)
        if keyword_count >= 2:  # If at least 2 keywords are found
            results[disease] = 'High'
        elif keyword_count == 1:  # If only 1 keyword is found
            results[disease] = 'Medium'
        else:
            results[disease] = 'Low'
    
    return results"""

@app.route('/analyze_report', methods=['POST'])
@login_required
def analyze_report():
    if 'medical_report' not in request.files:
        return redirect(url_for('report_analysis'))
    
    file = request.files['medical_report']
    file_ext = file.filename.split(".")[1].lower()
    print(file.filename,"\n",file_ext)
    if file.filename == '':
        return redirect(url_for('report_analysis'))
    if file_ext in ['.xlsx', '.xls']:
        text = extract_text_from_excel(file)
    elif file_ext == '.csv':
        text = extract_text_from_csv(file)
    """elif file_ext == '.txt':
        text = file.read().decode('utf-8')"""
    
    # Get file extension
    
    
    try:
        # Extract text based on file type
        
        # Save uploaded file temporarily

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_pdf:
            temp_pdf.write(file.read())
            temp_path = temp_pdf.name
        print("temp_path",temp_path,file_ext)
        # Extract text from PDF
        text = ""
        if file_ext == "pdf":
            with pdfplumber.open(temp_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        elif file_ext in IMAGE_FORMATS:
            img = Image.open(temp_path)
            text = pytesseract.image_to_string(img)

        # Clean up temp file
        os.unlink(temp_path)

        if not text.strip():
            return {
                "status_code":400,
                "error": "No text could be extracted from the PDF"
            }
        print("text",text)
        # Send to Gemini
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt+text
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.8,
                "topK": 40,
                "maxOutputTokens": 1024
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }

        # Set headers
        headers = {
            "Content-Type": "application/json"
        }

        # Make the API request
        response = requests.post(url, json=payload, headers=headers)
        print("response",response)
        if response.status_code == 200:
            print("response",response)
            response_data = response.json()
            print("response_data",response_data)
            try:
                # Extract the generated text from the response
                generated_text = response_data["candidates"][0]["content"]["parts"][0]["text"]
                print(generated_text)
                generated_text = markdown(generated_text)
                return render_template('report_analysis copy.html', results=generated_text)
            except (KeyError, IndexError) as e:
                return jsonify({"Error parsing response:": str(e),"Response:": {response_data}})
        else:
            return jsonify({"Error:": response.status_code+response.text})
    except Exception as e:
        return jsonify({"error":str(e)})

        
        
        """# Preprocess the text
        processed_text = preprocess_text(text)
        
        # Analyze the text for diseases
        results = analyze_text_for_diseases(processed_text)
        """
        
    
    except Exception as e:
        return jsonify({f"Error processing file:" :str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)