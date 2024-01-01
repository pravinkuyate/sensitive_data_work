import spacy
from flask import Flask, request, jsonify
from flask_sslify import SSLify

app = Flask(__name__)
sslify = SSLify(app)

# Use new trained saved model
model_dir = r"C:\Users\pravinkuyate\OneDrive\Desktop\project01"
print("Loading trained model from:", model_dir)
nlp2 = spacy.load(model_dir)

@app.route('/SensitiveInformation', methods=['POST'])
def validate_users():
    try:
        data = request.get_json()
        if 'Sentence' not in data:
            return jsonify({"error": "Missing 'Sentence' in the request body"}), 400

        result = nlp2(data['Sentence'])
        my_dict = {}

        for token in result:
            print(token, token.ent_type_)
            if token.is_title:
                my_dict[token.text] = token.ent_type_

        print(my_dict)
        return jsonify(my_dict), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)