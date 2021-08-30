import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def read_data():
    names= open(r'D:\Downloads\dinos.txt','r').read().lower()
    chars = list(set(names))
    names_array = names.split('\n')
    name_char_array = []
    for single_name in names_array:
        single_name_chars = [c for c in single_name]
        name_char_array.append(single_name_chars)
    chars.append('#')
    chars = sorted(chars)
    return chars,name_char_array

chars, name_char_array = read_data()
sequence_length = 5

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

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(128, activation='relu'),
    tf.keras.layers.Dense(total_chars, activation='softmax')
])
model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
model.fit(xs,ys,epochs=100)
model.save('dino.h5')

def predict_names(model,sequence_length=3,start_string=""):
    endOfFile = 0
    input_string = start_string[-sequence_length:]
    pad_length = sequence_length - len(input_string)
    if(pad_length>0):
        input_string = '#'*pad_length + input_string
    input_model = [char_to_ix[c] for c in input_string]
    output = []
    predicted_out = 1
    while predicted_out!=endOfFile:
        predicted_out = model.predict(np.array(input_model).reshape(-1,sequence_length,1))
        predicted_out = np.random.choice(28, p=predicted_out.ravel())
        input_model = input_model[1:5] + [predicted_out]
        if(predicted_out!=endOfFile):
            output.append(predicted_out)
    output = [ix_to_char[c] for c in output]
    return start_string + "".join(output)

for i in range(5):
    name_start = "b"
    output = predict_names(model, sequence_length, name_start.lower())
    print(output)