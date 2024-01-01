import spacy
import random
import csv
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training.example import Example

# Define output folder to save the new model
model_dir = r"C:\Users\pravinkuyate\OneDrive\Desktop\project01"

# Function to read data from CSV file
def read_data_from_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = []
        for row in reader:
            entities = {}
            for column, value in row.items():
                if column.lower() != 'text' and value:
                    entities[column.upper()] = [(0, len(value), column.upper())]
            data.append((row['TEXT'], {"entities": entities}))
    return data

# Train new NER model
def train_new_NER(model=None, output_dir=model_dir, n_iter=100, csv_file_path=None):
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

    # Read training data from CSV file
    if csv_file_path:
        TRAIN_DATA = read_data_from_csv(csv_file_path)

    # Add labels
    for _, annotations in TRAIN_DATA:
        for ent_type, spans in annotations.get("entities").items():
            for span in spans:
                ner.add_label(span[2])

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

# Specify the path to your CSV file
csv_file_path = r"C:\Users\pravinkuyate\OneDrive\Desktop\project01\sensitive_dataset.csv"

# Finally, train the model by calling the function
train_new_NER(csv_file_path=csv_file_path)
