'''
A checkear qué hacemos con este script.
'''


# TODO: 
#   - merge examLoader class with existing data storage (maybe huggingface datasets??).
#   - add methods for handling exams by metadata (maybe for language, for difficulty...).
#   - add methods for plotting results.
#   - add methods for storing results somewhere. It should happen whenever model.predict is called.

#----------------------------------------------------------------------------------------------------------------------------------
# examLoader reads only 1 exam.
# examProcesser loads the entire dataset of exams using examLoader for loading each.
#----------------------------------------------------------------------------------------------------------------------------------

import os
import json
from replier import Replier

class examLoader:
    def __init__(self):
        self.data = None

    def load_exam(self, filepath):
        with open(filepath, 'r') as file:
            self.data = json.load(file)

    def get_question(self, index):
        if self.data and 0 <= index < len(self.data):
            return self.data[index]
        else:
            raise IndexError("Question index out of range.")

    def get_all_questions(self):
        return self.data
    

class examProcessor:
    def __init__(self, folder_path):
        # Folder path is the folder where the exams lie.
        self.folder_path = folder_path

        # Exam loaders is a list of tuples (filename, examLoader). Maybe we should rethink this if exams are not stored in a 
        # folder within the repo.
        self.exam_loaders = []

        # Load all exams from storage.
        self.load_exams()
        
        #self.exam_results_path??

    def load_exams(self):
        """Iterates through the folder and initializes an examLoader for each file."""
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.json'):  
                filepath = os.path.join(self.folder_path, filename)
                loader = examLoader()
                loader.load_exam(filepath)
                self.exam_loaders.append((filename, loader))

    def evaluate_model(self, model: Replier):
        """Evaluates the model on all loaded exams."""
        results = {}
        for filename, loader in self.exam_loaders:
            questions = loader.get_all_questions()
            accuracy = self._evaluate_single_exam(model, questions, filename)
            results[filename] = accuracy
        return results

    def _evaluate_single_exam(self, model: Replier, questions: list[dict], file_path: str):
        """Evaluates a model on a single exam."""
        correct = 0
        total = len(questions)
        for index, question in enumerate(questions):
            predicted_answer = model.predict_zero_shot(question["question"], question["options"], index, file_path)
            if predicted_answer == question["answer"]:
                correct += 1
        return correct / total if total > 0 else 0

