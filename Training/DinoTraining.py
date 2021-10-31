import numpy as np
import tensorflow as tf
import mlflow
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\mheme\dino-name-generator-b95c587f2fc6.json'

def read_data():
    with open(r'D:\Downloads\dinos.txt','r') as names:
        names = names.read().lower()
        names_array = names.split('\n')
        name_char_array = []
        for single_name in names_array:
            single_name_chars = [c for c in single_name]
            name_char_array.append(single_name_chars)
    chars = list(set(names))
    chars.append('#')
    chars = sorted(chars)
    return chars,name_char_array

chars, name_char_array = read_data()
sequence_length = 3

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
total_chars = len(chars)

def get_data(name_char_array, sequence_length=3):
    train_x = []
    train_y = []
    for name in name_char_array:
        name.append('\n')# End of name character
        name.insert(0,'#')
        name = [char_to_ix[chars] for chars in name]
        for j in range(len(name)-sequence_length):
            train_x.append([name[j:j+sequence_length]])
            train_y.append([name[j+sequence_length]])
    train_x = np.array(train_x).reshape(-1,sequence_length,1).astype(np.float32)
    train_y = tf.keras.utils.to_categorical(train_y, num_classes=total_chars)
    return train_x, train_y

xs, ys = get_data(name_char_array, sequence_length)
#Required only for sparse_categorical_crossentropy
#train_y = np.array(train_y).reshape(-1,1,1)

cloudUri = 'gs://dino-name-generator-mlflow-artifacts'

if mlflow.get_experiment_by_name('dino-name-generator') is None:
    mlflow.create_experiment('dino-name-generator', artifact_location=cloudUri)
mlflow.set_experiment('dino-name-generator')
mlflow.tensorflow.autolog(every_n_iter=20)

with mlflow.start_run():
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, activation='relu',return_sequences=True),
        tf.keras.layers.LSTM(64, activation='relu'),
        tf.keras.layers.Dense(total_chars, activation='softmax')
    ])
    model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
    model.fit(xs,ys,epochs=40)

mlflow.log_param("char_num_mapping",ix_to_char)

print(mlflow.get_tracking_uri())