import pandas as pd
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import ModelCheckpoint

# --- Parameters ---
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 64

# --- 1. Load Data ---
print("Loading data...")
df = pd.read_csv('urls.csv')
urls = df['url'].values
labels = df['label'].values

# --- 2. Preprocess Data ---
print("Preprocessing data...")
# Initialize and fit tokenizer
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, char_level=True, oov_token='<UNK>')
tokenizer.fit_on_texts(urls)

# Convert URLs to sequences of characters
sequences = tokenizer.texts_to_sequences(urls)
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

# Split data (we use a small test split here, in real research this would be larger)
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# --- 3. Build LSTM Model ---
print("Building model...")
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1,
                    output_dim=EMBEDDING_DIM,
                    input_length=MAX_SEQUENCE_LENGTH))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid')) # Sigmoid for binary classification

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# --- 4. Train Model ---
print("Training model...")
# Note: With our tiny dataset, this will be instant and overfit.
# In your real project, you'd have thousands of epochs and much more data.
model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))

# --- 5. Save Model and Tokenizer ---
print("Saving model and tokenizer...")
model.save('url_model.h5')

# Save the tokenizer for the app
tokenizer_json = tokenizer.to_json()
with open('url_tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

print("URL model training complete.")
