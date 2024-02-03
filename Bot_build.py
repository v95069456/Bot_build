from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from spellchecker import SpellChecker


def correct_spelling(text):
    spell = SpellChecker()

    # Разделяем текст на слова
    words = text.split()

    # Проверяем каждое слово на орфографические ошибки
    corrected_words = [spell.correction(word) if spell.correction(word) is not None else word for word in words]

    # Собираем исправленный текст
    corrected_text = ' '.join(corrected_words)

    return corrected_text

class Text_Chatbot:
    def __init__(self, name, file_path, eror):
        self.name = name
        self.eror = eror
        self.file_path = file_path
        self.load_responses()

    def load_responses(self):
        with open(self.file_path, 'r') as file:
            self.responses = file.read().splitlines()

        self.vectorizer = TfidfVectorizer()
        self.response_vectors = self.vectorizer.fit_transform(self.responses)

    def preprocess_text(self, text):
        text = text.lower()
        # Add more preprocessing steps if needed
        return text

    def add_response(self, new_response):
        # Add a new response to the list
        self.responses.append(new_response)
        
        # Update the TF-IDF vectors
        self.response_vectors = self.vectorizer.fit_transform(self.responses)

    def get_response(self, user_input):
        user_input = self.preprocess_text(user_input)
        user_vector = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, self.response_vectors).flatten()

        # Handle case when there is no matching response
        if similarities.max() == 0:
            return self.eror

        best_response_index = similarities.argmax()
        return f'{self.name} --> ' + self.responses[best_response_index]
#########################################################################################################################################################################
class Pairs_Chatbot:
    def __init__(self, name, pairs, reflections=None, default_response="I'm sorry, I don't understand.", error_response="An error occurred. Please try again."):
        self.name = name
        self.pairs = pairs
        self.reflections = reflections if reflections else {}
        self.default_response = default_response
        self.error_response = error_response

    def preprocess_text(self, text):
        # Add more preprocessing steps if needed
        return text.lower()

    def transform_input(self, text):
        words = text.split()
        for i in range(len(words)):
            if words[i] in self.reflections:
                words[i] = self.reflections[words[i]]
        return ' '.join(words)

    def respond(self, user_input):
        try:
            user_input = self.preprocess_text(user_input)
            user_input = self.transform_input(user_input)

            for pattern, response in self.pairs:
                match = re.search(pattern, user_input)
                if match:
                    return response

            return self.default_response
        except Exception as e:
            print(f"Error: {e}")
            return self.error_response
