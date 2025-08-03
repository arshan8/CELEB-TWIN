

# 5. Predict
test_comment = ["very bad"]
test_seq = tokenizer.texts_to_sequences(test_comment)
test_pad = pad_sequences(test_seq, maxlen=3)
pred = model.predict(test_pad, verbose=0)
print("Sentiment score (0=neg, 1=pos):", pred[0][0])








import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

# 1. Data
comments = ["bad shit", "very good", "worst ever", "amazing", "shit movie", "awesome experience"]
y = np.array([0, 1, 0, 1, 0, 1])  # 0 = negative, 1 = positive

# 2. Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(comments)
sequences = tokenizer.texts_to_sequences(comments)
X = pad_sequences(sequences, maxlen=3)

# 3. Model
model = Sequential([
    Embedding(input_dim=50, output_dim=8, input_length=3),
    LSTM(10),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train
model.fit(X, y, epochs=100, verbose=0)

# 5. Predict and Report
pred_probs = model.predict(X, verbose=0)
pred_labels = (pred_probs > 0.5).astype(int).flatten()

print(classification_report(y, pred_labels, target_names=['Negative', 'Positive']))








import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

# 1. Sample data
comments = ["bad shit", "very good", "amazing", "worst ever", "not bad", "excellent", "sucks", "loved it"]
labels = [0, 1, 1, 0, 1, 1, 0, 1]  # 0 = negative, 1 = positive

# 2. Tokenize and pad
tokenizer = Tokenizer()
tokenizer.fit_on_texts(comments)
sequences = tokenizer.texts_to_sequences(comments)
padded = pad_sequences(sequences, padding='post')

# 3. Define model
vocab_size = len(tokenizer.word_index) + 1
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=8, input_length=padded.shape[1]),
    SimpleRNN(16),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. Train
model.fit(padded, np.array(labels), epochs=20, verbose=0)

# 5. Predict and report
preds = (model.predict(padded) > 0.5).astype("int32").flatten()
print("Classification Report:\n")
print(classification_report(labels, preds))




















from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
import numpy as np

# 1. Sample data
comments = ["bad shit", "very good", "amazing", "worst ever", "not bad", "excellent", "sucks", "loved it"]
labels = [0, 1, 1, 0, 1, 1, 0, 1]  # 0 = negative, 1 = positive

# 2. Tokenize and pad
tokenizer = Tokenizer()
tokenizer.fit_on_texts(comments)
sequences = tokenizer.texts_to_sequences(comments)
padded = pad_sequences(sequences, padding='post')

# 3. Define CNN model
vocab_size = len(tokenizer.word_index) + 1
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=8, input_length=padded.shape[1]),
    Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'),  # <== fixed here
    GlobalMaxPooling1D(),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. Train
model.fit(padded, np.array(labels), epochs=20, verbose=0)

# 5. Predict and report
preds = (model.predict(padded) > 0.5).astype("int32").flatten()
print("Classification Report:\n")
print(classification_report(labels, preds))










import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

# 1. Sample dataset
texts = ["worst movie", "awesome experience", "not good", "loved it", "bad", "fantastic", "hate it", "superb"]
labels = [0, 1, 0, 1, 0, 1, 0, 1]  # 0 = negative, 1 = positive

# 2. Tokenization and padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
seqs = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(seqs, padding='post')

# 3. Define GRU model
vocab_size = len(tokenizer.word_index) + 1
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=8, input_length=padded.shape[1]),
    GRU(16),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. Train
model.fit(padded, np.array(labels), epochs=20, verbose=0)

# 5. Evaluate
preds = (model.predict(padded) > 0.5).astype("int32").flatten()
print("Classification Report:\n")
print(classification_report(labels, preds))










