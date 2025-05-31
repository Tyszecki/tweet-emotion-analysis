from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

def create_model(vocab_size, max_length, num_classes):
    """Tworzy model LSTM do klasyfikacji emocji"""
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_length),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

def prepare_data(df):
    """Przygotowuje dane do trenowania"""
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(df['clean_text'])
    
    sequences = tokenizer.texts_to_sequences(df['clean_text'])
    padded = pad_sequences(sequences, maxlen=100)
    
    le = LabelEncoder()
    labels = le.fit_transform(df['label'])
    
    return padded, labels, tokenizer, le
