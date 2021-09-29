import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sn


BATCH_SIZE = 32
EPOCHS = 10
MAX_LEN = 75
EMBEDDING = 20

data = pd.read_csv("ner_dataset.csv", encoding = "ISO-8859-1")
data = data.fillna(method="ffill")
print("Number of sentences: ", len(data.groupby(['Sentence #'])))
words = list(set(data["Word"].values))
n_words = len(words)
print("Number of words in the dataset: ", n_words)
tags = list(set(data["Tag"].values))
print("Tags:", tags)
n_tags = len(tags)
print("Number of Labels: ", n_tags)

def get_sentences(data):
    agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
    grouped = data.groupby("Sentence #").apply(agg_func)
    sentences = [s for s in grouped]
    return sentences

sentences = get_sentences(data)

# Dictionaries
word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx["UNK"] = 1 # Unknown words
word2idx["PAD"] = 0 # Padding

idx2word = {i: w for w, i in word2idx.items()}

tag2idx = {t: i+1 for i, t in enumerate(tags)}
tag2idx["PAD"] = 0

idx2tag = {i: w for w, i in tag2idx.items()}

X = [[word2idx[w[0]] for w in s] for s in sentences]

# Padding each sentence to have the same length
X = pad_sequences(maxlen=MAX_LEN, sequences=X, padding="post", value=word2idx["PAD"])

y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(maxlen=MAX_LEN, sequences=y, padding="post", value=tag2idx["PAD"])

# One-Hot encoding
y = [np.eye(n_tags + 1)[i] for i in y]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)
print(X_tr.shape), print(X_te.shape), print(np.array(y_tr).shape), print(np.array(y_te).shape)

model = keras.Sequential()
model.add(layers.Embedding(input_dim=n_words+2, output_dim=EMBEDDING, # n_words + 2 (PAD & UNK)
                  input_length=MAX_LEN))
model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True), input_shape=(800,13)))
model.add(layers.Dropout(0.1))
model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True), input_shape=(800,128)))
model.add(layers.Dropout(0.1))
model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True), input_shape=(800,128)))
model.add(layers.Dropout(0.1))
model.add(layers.TimeDistributed(layers.Dense(64, activation="relu")))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(n_tags+1, activation="softmax"))
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_tr, np.array(y_tr), batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=1)

# Testing the model on test data
pred_cat = model.predict(X_te)
pred = np.argmax(pred_cat, axis=-1)
print(pred)
y_te_true = np.argmax(y_te, -1)
print(pred.shape)
print(y_te_true)
print(np.array(y_te_true).shape)

# Converting the index to tag
pred_tag = []
for row in pred:
  for i in row:
    pred_tag.append(idx2tag[i])

y_te_true_tag = []
for row in y_te_true:
  for i in row:
    y_te_true_tag.append(idx2tag[i])

# print(pred_tag)
# print(" ")
# print(" ")
# print(" ")
# print(y_te_true_tag)
report = classification_report(y_pred=pred_tag, y_true=y_te_true_tag)
print(report)

# choose a random number between 0 and len(X_te)
i = np.random.randint(0,X_te.shape[0]) 
p = model.predict(np.array([X_te[i]]))
p = np.argmax(p, axis=-1)
true = np.argmax(y_te[i], -1)

print("Sample number {} of {} (Test Set)".format(i, X_te.shape[0]))

# Output
print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
print(30 * "=")
for w, t, pred in zip(X_te[i], true, p[0]):
    if w != 0:
        print("{:15}: {:5} {}".format(words[w-2], idx2tag[t], idx2tag[pred]))

cnf_matrix = confusion_matrix(y_te_true_tag, pred_tag)
print(cnf_matrix)

list2 = ["PAD"]
# print(len(list2))
padded_tags = tags + list2
# print(padded_tags)
padded_tags.sort()
print(padded_tags)
# print(len(tags))
# print(len(padded_tags))

df_cm = pd.DataFrame(cnf_matrix, index = [i for i in padded_tags],
                  columns = [i for i in padded_tags])
plt.figure(figsize = (16,12))
# sn.set(font_scale=0.8) # for label size
sn.heatmap(df_cm, annot=True)

# sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
# plt.show()
# plt.matshow(cnf_matrix)
# plt.show()
