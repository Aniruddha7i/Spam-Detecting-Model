nltk.download()
import nltk
from flask import Flask, render_template, jsonify, request
import pickle
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# text to formed text
stopWords = nltk.corpus.stopwords.words('english')


def transform_text(text):
    #convart lower case
    text = text.lower()

    # tokenization :: convert string in to a list of words
    text = nltk.word_tokenize(text)

    # remove spatial char
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]  # this is cloning
    y.clear()

    # remove stop words and punctuation
    for i in text:
        if i not in stopWords and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # stemming
    # stemming is process of reducing a word to its root word
    # for example danceing ---> dance
    ps = nltk.porter.PorterStemmer()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


app: Flask = Flask(__name__, template_folder='template', static_folder='static')

# Open the file in read binary mode ('rb')
with open('model.pkl', 'rb') as f:
    # Load the pickled object from the file
    detector = pickle.load(f)

with open('ifid.pkl', 'rb') as f2:
    # Load the pickled object from the file
    ifid = pickle.load(f2)


@app.route("/", methods=["GET"])
def home_temp():
    return render_template('home.html')


@app.route('/submit', methods=["POST"])
def submit_form():
    mess = request.form.get("mail_body")
    mess_pred = transform_text(mess)
    # different vectorizer :: TfidfVectorizer
    mp = [mess_pred]
    x = ifid.transform(mp).toarray()
    result = detector.predict(x)
    if result.any() == 0:
        reply = "Successfully submitted!!"
        spam = False
        return render_template('home.html', is_spam=spam, reply=reply)

    else:
        reply = "This is a Spam"
        return render_template('home.html', is_spam=True, reply=reply)
    # age = request.form.get("int", "age")  # Handle missing values
    # return f"Hello, {name}! You are {age} years old."


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
