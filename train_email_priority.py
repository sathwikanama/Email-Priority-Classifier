# train_email_priority.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import pickle

# ---------- 1. Load dataset ----------
# Your CSV should have columns: 'subject', 'body', 'label' (1=Important, 0=Not Important)
data = pd.read_csv("enron_email_subset.csv")

# Combine subject + body
data['text'] = data['subject'].fillna('') + " " + data['body'].fillna('')

X = data['text']
y = data['label']

# ---------- 2. Split dataset ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- 3. TF-IDF Vectorization ----------
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# Save vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# ---------- 4. Build Neural Network ----------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_dim=X_train_vec.shape[1], activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ---------- 5. Train model ----------
model.fit(X_train_vec, y_train, epochs=5, batch_size=16, validation_split=0.1)

# ---------- 6. Evaluate ----------
loss, acc = model.evaluate(X_test_vec, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# ---------- 7. Save model ----------
model.save("email_priority_model.keras")
print("✅ Training complete. Model & vectorizer saved.")
