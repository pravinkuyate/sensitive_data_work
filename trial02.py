from flask import Flask, render_template, request
import re
import pickle

app = Flask(__name__)

patterns = {
    'Email': r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    'credit_card': r'\b(?:\d[ -]*?){13,16}\b',
    'contact': r'(\+\d{1,4}\s*)?(\d{10}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\(\d{3}\)\s*\d{3}[-.\s]?\d{4})',
    'pan': r'[A-Za-z]{5}[0-9]{4}[A-Za-z]{1}',
    'aadhar': r'\b\d{4}\s\d{4}\s\d{4}\b',
    'password': r'(?i)\bpassword\b\s*:\s*([^\s]+)'
}

def detect_sensitive_data(text):
    sensitive_info = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            sensitive_info[key] = matches
    return sensitive_info

@app.route('/', methods=['GET', 'POST'])
def index():
    sensitive_data = None

    if request.method == 'POST':
        user_input = request.form.get('user_input', '')
        sensitive_data = detect_sensitive_data(user_input)

        # Save sensitive data to pickle file (adjust the file path as needed)
        arjun_model = "arjun_model.pkl"
        with open(arjun_model, 'wb') as file:
            pickle.dump(sensitive_data, file)

    return render_template('index.html', sensitive_data=sensitive_data)

if __name__ == '__main__':
    app.run(debug=True)
