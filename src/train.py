import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.model import create_model, prepare_data
import matplotlib.pyplot as plt
import pickle

def train():
    # 1. Wczytanie danych
    df = load_data('data/tweets.csv')
    
    # 2. Przygotowanie danych
    X, y, tokenizer, label_encoder = prepare_data(df)
    
    # 3. Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 4. Budowa modelu
    model = create_model(vocab_size=10000, max_length=100, num_classes=len(label_encoder.classes_))
    
    # 5. Trenowanie
    history = model.fit(X_train, y_train,
                       epochs=15,
                       batch_size=64,
                       validation_data=(X_test, y_test))
    
    # 6. Zapis modelu i narzędzi
    model.save('models/emotion_model.h5')
    with open('models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('models/label_encoder.pickle', 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # 7. Wizualizacja wyników
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('results/accuracy_plot.png')
    plt.close()

if __name__ == "__main__":
    train()
