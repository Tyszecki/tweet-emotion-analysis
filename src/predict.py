import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_emotion(text):
    """Przewiduje emocję na podstawie tekstu"""
    # Wczytanie modelu i narzędzi
    model = load_model('models/emotion_model.h5')
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('models/label_encoder.pickle', 'rb') as handle:
        le = pickle.load(handle)
    
    # Przetwarzanie tekstu
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100)
    
    # Predykcja
    prediction = model.predict(padded)
    return le.inverse_transform([np.argmax(prediction)])[0]

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(f"Predicted emotion: {predict_emotion(sys.argv[1])}")
    else:
        print("Please provide a text to analyze")
