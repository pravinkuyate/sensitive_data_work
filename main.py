from flask import Flask, request, jsonify
import pickle
app = Flask(__name__)

@app.route('/')
def index():
    return 'Welcome to the arjun and jaydeeps Classification API!'

if __name__ == '__main__':
    app.run(debug=True)




with open('arjun_model', 'rb') as file:
    model = pickle.load(file)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']

    # Perform prediction using the loaded model
    prediction = model.predict([text])[0]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)


