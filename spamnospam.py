import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard

# Create a TensorBoard callback
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

# Load dataset
df = pd.read_csv("SPAM text message 20170820 - Data.csv", index_col=None)

# Preprocess data
df_spam = df[df['Category'] == 'spam']
df_ham = df[df['Category'] == 'ham']
df_ham_down_sample = df_ham.sample(df_spam.shape[0])
df_balanced = pd.concat([df_spam, df_ham_down_sample])
df_balanced['Category'] = df['Category'].map({'spam': 1, 'ham': 0})

sentences = df_balanced["Message"].to_numpy()
labels = df_balanced["Category"].to_numpy()
sentence_len = [len(sentence.split()) for sentence in sentences]
average_sentence_len = round(sum(sentence_len) / len(sentence_len))
np.percentile(sentence_len, 95), average_sentence_len
max_tokens = 10000
max_length = 32
text_vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens,
                                                   output_sequence_length=max_length,
                                                   standardize="lower_and_strip_punctuation",
                                                   split="whitespace",
                                                   output_mode="int")
text_vectorizer.adapt(sentences)

# Build neural network model
universal_sentence_encoder = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder-large/5',
                                           input_shape=[],
                                           dtype=tf.string,
                                           trainable=False,
                                           name="USE")

model = tf.keras.models.Sequential([
    universal_sentence_encoder,
    tf.keras.layers.Dense(units=128, activation="relu"),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(units=32, activation="relu"),
    tf.keras.layers.Dense(units=1, activation="sigmoid")
], name="universal_sentence_encoder")

# Compile the model
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=["accuracy"])

# Train the model
history = model.fit(x=sentences,
                    y=labels,
                    epochs=10,
                    validation_split=0.2,
                    callbacks=[tensorboard_callback])

# Save the model to a file (.h5)
model.save('spam_classifier_model.h5', include_optimizer=True)

# Load the saved model with custom objects
loaded_model = tf.keras.models.load_model('spam_classifier_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})

# Evaluate the model
test_loss, test_acc = loaded_model.evaluate(sentences, labels)
print(f'Test accuracy: {test_acc}')

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Save the plot as an image file
plt.savefig('training_history_plot.png')

# Show the plot
plt.show()
