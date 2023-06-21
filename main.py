import time
from flask import Flask, send_file, request, jsonify, render_template, json, request, redirect, url_for, session, flash, Response, make_response
from flask_restx import Resource, Api, reqparse
from datetime import date, datetime

from flask_sqlalchemy import SQLAlchemy
from flask_mysqldb import MySQL, MySQLdb
import MySQLdb.cursors
import re
import bcrypt
from werkzeug.security import generate_password_hash

from flask_cors import  CORS
import json
from flask_marshmallow import Marshmallow
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
import os
import cv2 as cv
import numpy


import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import json
import random
from keras.models import load_model
model = load_model('static/assets/chatbot/chatbot_model.h5')
from chatbot import model, chatbot_response,getResponse, clean_up_sentence,predict_class,lemmatizer, get_bot_response
from chatbot import intents, words, classes


app = Flask(__name__)
app.config['SECRET_KEY'] = '$#fabillah31'

#untuk konfigurasi database
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'absen'
mysql = MySQL(app)

api = Api(app, title='Absensi Wajah MySQL', default='Input Data', default_label='wajah',
          description='Kelompok 6 : </br>'
                      'Buat Database : <a href="/create_db">Klik Buat</a></br>'
                      'History Database  : <a href="/all-data-absen"><input type=submit value=Lihat></a>'
          )
CORS(app)

# db.init_app(app)
app.config["SQLALCHEMY_DATABASE_URI"] = 'mysql://root:''@localhost/absen'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
ma = Marshmallow(app)

# fungsi_model_ai
#untuk memasang model
def live_frame():

    cap = cv.VideoCapture(0) #'http://192.168.1.114:8080/video')
    i = 0
    model = load_model('assets/keras_model.h5')

    # Grab the labels from the labels.txt file. This will be used later.
    np.set_printoptions(suppress=True)

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    a = 0
    person = ""
    akurasi = ""
    while True:
        _,img = cap.read()
        image = cv.resize(img, (224, 224))

        image = np.asarray(image)

        normalize_image_array = (image.astype(np.float32) / 127.0) - 1

        data[0] = normalize_image_array

        prediction = model.predict(data)

        for i in prediction:
            if i[0] > 0.75:
                person = "Iskandar"
                akurasi = str(i[0])
            if i[1] > 0.75:
                person = "Fabil"
                akurasi = str(i[1])
            if i[2] > 0.75:
                person = "Darto"
                akurasi = str(i[2])
            if i[3] > 0.75:
                person = "Rizki"
                akurasi = str(i[3])
            a = a + 1
            filename = "C:/Users/User/Pictures/Saved Pictures/" + person + str(a) + ".png"
            if a < 3:
                cv.imwrite(filename, img)
                time.sleep(1)
            desc = "Nama : "+person
            acc = "Akurasi : "+akurasi
            print(person)
            print(akurasi)
            img = cv.resize(image, (1920,1200))
            cv.putText(img, desc, (10, 70), cv.FONT_HERSHEY_COMPLEX, 3,
                               (200, 200, 0), 2)
            cv.putText(img, acc, (10, 200), cv.FONT_HERSHEY_COMPLEX, 3,
                               (200, 200, 0), 2)
        frame = cv.imencode('.jpg', img)[1]
        encode = frame.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + encode + b'\r\n')
        time.sleep(0.1)





# model database
class Absensi(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    nama = db.Column(db.String(100), nullable=False)
    created_date = db.Column(db.TIMESTAMP, nullable=False)

# inisialisasi/validasi database
    def __init__(self, filename, nama, created_date):
        # self.id = id
        self.filename = filename
        self.nama = nama
        self.created_date = created_date

class AbsensiSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Absensi
        load_instance = True

# membuat fungsi create db
@app.route("/create_db", methods=["GET"])
def create_db():
    with app.app_context():
        db.create_all()
        return "Database Telah dibuat" + ' <a href="/"> Kembali</a>'

# membuat fungsi log db
@app.route("/all-data-absen", methods=["GET"])
def getAllAbsensi():
    history = Absensi.query.all()
    absensi_schema = AbsensiSchema(many=True)
    output = absensi_schema.dump(history)
    return jsonify({'History': output})

app.config['UPLOAD_WAJAH'] = 'assets/wajah'

parser4Param = reqparse.RequestParser()
parser4Param.add_argument('file', location='files', help='Filename', type=FileStorage, required=True)
# parser4Param.add_argument('nama', type=int, help='nama', location='args')


parser4Body = reqparse.RequestParser()
parser4Body.add_argument('image', location='files', help='Image', type=FileStorage, required=True)
# parser4Body.add_argument('nama', type=int, help='nama', location='args')

# swagger login
loginparser =  reqparse.RequestParser()
loginparser.add_argument('email', type=str, help='email', location='form' )
loginparser.add_argument('password', type=str, help='password', location='form')
@api.route('/loginapi', methods=['POST','GET'])
class loginUser(Resource):
    @api.expect(loginparser)
    def post(self):
        args = loginparser.parse_args()
        email = args['email']
        password = args['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE email = %s AND password = %s', (email, password,))
        # Fetch one record and return result
        users = cursor.fetchone()
        # If account exists in accounts table in out database
        if users:
            return jsonify({'message': 'login Sukses', 'data':users})
        else:
            return make_response(jsonify({'message':'login gagal'}),400)

def create_train():
    pass
@api.route('/image/', methods=["GET", "POST"])
class AbsensiAPI(Resource):
    @api.expect(parser4Body)
    def post(self):
            args = parser4Body.parse_args()
            created_date = datetime.now()
            file = args['image']
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_WAJAH'], filename))

            image = cv.imread("assets/wajah/" + file.filename)

            # Load the model
            model = load_model('assets/keras_model.h5')

            # Grab the labels from the labels.txt file. This will be used later.
            np.set_printoptions(suppress=True)

            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

            person = ""
            image = cv.resize(image, (224, 224))

            image = np.asarray(image)

            normalize_image_array = (image.astype(np.float32) / 127.0) - 1

            data[0] = normalize_image_array

            prediction = model.predict(data)

            for i in prediction:
                if i[0] > 0.75:
                    person = "Iskandar"
                if i[1] > 0.75:
                    person = "Fabil"
                if i[2] > 0.75:
                    person = "Darto"
                if i[3] > 0.75:
                    person = "Rizki"

                print(person)

            absensi = Absensi(
                filename=filename,
                nama=person,
                created_date=created_date
            )
            db.session.add(absensi)
            db.session.commit()
            return {
                # 'nip': nip,
                'filename': filename,
                'nama':person,
                'status': 200,
                'message': f"Data dengan filename {person} Sudah Absen"
            }

    def get(self):
        Absens = db.session.execute(db.select(Absensi).order_by(Absensi.id)).scalars()
        history = []
        for absensi in Absens:
            date = absensi.created_date.strftime("%m/%d/%Y, %H:%M:%S")
            history.append({
                'id': absensi.id,
                'filename': absensi.filename,
                'created_date': date,
            })
        return history

# route
@app.route('/login/', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        # Create variables for easy access
        email = request.form['email']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE email = %s AND password = %s', (email, password,))
        # Fetch one record and return result
        users = cursor.fetchone()
        # If account exists in accounts table in out database
        if users:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = users['id']
            session['email'] = users['email']
            # Redirect to home page
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    return render_template('login.html')


@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    else:
        name = request.form['name']
        email = request.form['email']
        password = request.form['password'].encode('utf-8')

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (name,email,password) VALUES (%s,%s,%s)", (name, email,password))
        mysql.connection.commit()
        session['name'] = request.form['name']
        session['email'] = request.form['email']
        return redirect(url_for('login'))

@app.route('/home')
def home():
    title = 'Absensi | Home'
    return render_template('home.html', title=title)

@app.route('/absen', methods=["GET", "POST"])
def absen():
        title = 'Absensi | Absen'
        return render_template('absen.html', title=title)

@app.route('/history', methods=["GET"])
def history():
        title = 'Absensi | History'
        return render_template('history.html', title=title)

# route chatbot (menampung pertanyaan user)
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

intents = json.loads(open('static/assets/chatbot/intents.json').read())
words = pickle.load(open('static/assets/chatbot/words.pkl', 'rb'))
classes = pickle.load(open('static/assets/chatbot/classes.pkl', 'rb'))

# def clean_up_sentence(sentence):
#     # tokenize the pattern - split words into array
#     sentence_words = nltk.word_tokenize(sentence)
#     # stem each word - create short form for word
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words
#
# # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
#
# def bow(sentence, words, show_details=True):
#     # tokenize the pattern
#     sentence_words = clean_up_sentence(sentence)
#     # bag of words - matrix of N words, vocabulary matrix
#     bag = [0]*len(words)
#     for s in sentence_words:
#         for i,w in enumerate(words):
#             if w == s:
#                 # assign 1 if current word is in the vocabulary position
#                 bag[i] = 1
#                 if show_details:
#                     print ("found in bag: %s" % w)
#     return(np.array(bag))
#
# def predict_class(sentence, model):
#     # filter out predictions below a threshold
#     p = bow(sentence, words,show_details=False)
#     res = model.predict(np.array([p]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
#     # sort by strength of probability
#     results.sort(key=lambda x: x[1], reverse=True)
#     return_list = []
#     for r in results:
#         return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
#     return return_list
#
# def getResponse(ints, intents_json):
#     tag = ints[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if(i['tag']== tag):
#             result = random.choice(i['responses'])
#             break
#     return result
#
# def chatbot_response(msg):
#     ints = predict_class(msg, model)
#     res = getResponse(ints, intents)
#     return res




@app.route('/live')
def live():
    return Response(live_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # DEBUG is SET to TRUE. CHANGE FOR PROD
    app.run(host='192.168.43.161',debug=True)