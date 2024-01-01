





import re   
import pickle


patterns = { 
    'Email':r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    'credit_card': r'\b(?:\d[ -]*?){13,16}\b',
    'contact':r'(\+\d{1,4}\s*)?(\d{10}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\(\d{3}\)\s*\d{3}[-.\s]?\d{4})',
    'pan': r'[A-Za-z]{5}[0-9]{4}[A-Za-z]{1}',
    'aadhar': r'\b\d{4}\s\d{4}\s\d{4}\b',
    'password':r'(?i)\bpassword\b\s*:\s*([^\s]+)'
    
}


def detect_sensitive_data(text):
    sensitive_info = {}
    for key, pattern in patterns.items():
        
        matches = re.findall(pattern, text)
        print(matches)
        if matches:
            sensitive_info[key] = matches
    return sensitive_info


example_text ="hello inkpk9528j"

sensitive_data = detect_sensitive_data(example_text)

arjun_model=r"C:\Users\pravinkuyate\OneDrive\Desktop\project01\arjun_model.pkl"
    

with open(arjun_model, 'wb') as file:
    pickle.dump(sensitive_data, file)
    #print(file)
    print("Model Created")
    print(sensitive_data)

    



