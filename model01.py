import spacy
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training.example import Example

# Define output folder to save the new model
model_dir = r"C:\Users\pravinkuyate\OneDrive\Desktop\project01"

# Train new NER model
def train_new_NER(model=None, output_dir=model_dir, n_iter=100):
    """Load the model, set up the pipeline, and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load an existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create a blank Language class
        print("Created a blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Training examples
    TRAIN_DATA = [
        
       ("Amit Sharma's Aadhar is 9876-5432-1098 and PAN is MNOP5678C.", {"entities": [(0, 12, "NAME"), (18, 38, "AADHAR"), (51, 61, "PAN"), (70, 82, "CONTACT"), (87, 111, "CREDIT_CARD")]}),
    ("Neha Verma's Contact is +91 8765432101 and Credit Card is 2345-6789-0123-4567.", {"entities": [(0, 11, "NAME"), (24, 44, "CONTACT"), (65, 89, "CREDIT_CARD"), (99, 117, "AADHAR"), (121, 131, "PAN")]}),
    ("Rahul Kapoor,rahul567,6543-2109-8765,GHIJ4567K,+91 6789012345,3456-7890-1234-5678.", {"entities": [(0, 11, "NAME"), (17, 27, "PAN"), (40, 60, "CONTACT"), (75, 99, "CREDIT_CARD"), (110, 128, "AADHAR")]}),
    ("Kavya Reddy,kavya123,8901-2345-6789,LKJI7890E,+91 9876543210,4567-9012-3456-7890.", {"entities": [(0, 11, "NAME"), (17, 37, "AADHAR"), (50, 60, "PAN"), (75, 99, "CREDIT_CARD"), (109, 127, "CONTACT")]}),
    ("Vikram Singh,vikram789,2345-6789-0123,OPQR9012A,+91 5432109876,5678-9012-3456-7890.", {"entities": [(0, 12, "NAME"), (18, 38, "AADHAR"), (51, 61, "PAN"), (76, 100, "CREDIT_CARD"), (110, 128, "CONTACT")]}),
    ("Ananya Gupta,ananya567,5678-9012-3456,WXYZ1234P,+91 8765432109,6789-0123-4567-8901.", {"entities": [(0, 12, "NAME"), (18, 38, "AADHAR"), (51, 61, "PAN"), (76, 100, "CREDIT_CARD"), (111, 129, "CONTACT")]}),
    ("Rajat Verma,rajat123,7890-1234-5678,BCDE5678I,+91 7654321098,7890-1234-5678-9012.", {"entities": [(0, 10, "NAME"), (16, 36, "AADHAR"), (49, 59, "PAN"), (74, 98, "CREDIT_CARD"), (108, 126, "CONTACT")]}),
    ("Priya Yadav,priya789,1234-5678-9012,FGHI1234Q,+91 6543210987,8901-2345-6789-0123.", {"entities": [(0, 10, "NAME"), (16, 36, "AADHAR"), (49, 59, "PAN"), (74, 98, "CREDIT_CARD"), (108, 126, "CONTACT")]}),
    ("Shubham Patel,shubham567,4567-8901-2345,JIKL9012R,+91 5432109876,9012-3456-7890-1234.", {"entities": [(0, 13, "NAME"), (19, 39, "AADHAR"), (52, 62, "PAN"), (77, 101, "CREDIT_CARD"), (111, 129, "CONTACT")]}),
    ("Tanvi Kumar,tanvi123,6789-0123-4567,LMNO4567S,+91 9876543210,0123-4567-8901-2345.", {"entities": [(0, 11, "NAME"), (17, 37, "AADHAR"), (50, 60, "PAN"), (75, 99, "CREDIT_CARD"), (109, 127, "CONTACT")]}),
    ("Mayank Sharma,mayank789,8901-2345-6789,HGFE5678T,+91 8765432109,1234-5678-9012-3456.", {"entities": [(0, 12, "NAME"), (18, 38, "AADHAR"), (51, 61, "PAN"), (76, 100, "CREDIT_CARD"), (110, 128, "CONTACT")]}),
    ("Anjali Reddy,anjali567,2345-6789-0123,IJKL9012U,+91 7654321098,2345-6789-0123-4567.", {"entities": [(0, 12, "NAME"), (18, 38, "AADHAR"), (51, 61, "PAN"), (76, 100, "CREDIT_CARD"), (111, 129, "CONTACT")]}),
    ("Arjun Gupta,arjun123,5678-9012-3456,PQRS1234V,+91 6543210987,3456-7890-1234-5678.", {"entities": [(0, 11, "NAME"), (17, 37, "AADHAR"), (50, 60, "PAN"), (75, 99, "CREDIT_CARD"), (109, 127, "CONTACT")]}),
    ("Sanya Verma,sanya789,7890-1234-5678,IJKL5678W,+91 5432109876,4567-9012-3456-7890.", {"entities": [(0, 11, "NAME"), (17, 37, "AADHAR"), (50, 60, "PAN"), (75, 99, "CREDIT_CARD"), (110, 128, "CONTACT")]}),
    ("Rajesh Kumar,rajesh567,5678-9012-3456,WXYZ1234X,+91 9876543210,5678-9012-3456-7890.", {"entities": [(0, 12, "NAME"), (18, 38, "AADHAR"), (51, 61, "PAN"), (76, 100, "CREDIT_CARD"), (110, 128, "CONTACT")]}),
    







    ]

    # Add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                example = []
                for i in range(len(texts)):
                    doc = nlp.make_doc(texts[i])
                    example.append(Example.from_dict(doc, annotations[i]))

                # Update the model
                nlp.update(example, drop=0.5, losses=losses)

    # save model to the output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

# Finally, train the model by calling the function
train_new_NER()
