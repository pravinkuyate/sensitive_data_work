import re
import pickle



def detect_sensitive_data(text, model):
    sensitive_info = {}
    for key, patterns in model.items:
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                if key not in sensitive_info:
                    sensitive_info[key] = []
                sensitive_info[key].extend(matches)
    return sensitive_info

def main():
    
    with open('sensitive_data_model.pkl', 'rb') as file:
        sensitive_data_model = pickle.load(file)

   
    user_input = input("Enter text to check for sensitive information: ")

    
    detected_sensitive_data = detect_sensitive_data(user_input, sensitive_data_model)

    # Print the detected sensitive data in JSON format
    print(detected_sensitive_data)

if __name__ == "__main__":
    main()
