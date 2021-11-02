import tensorflow as tf
import numpy as np
import mlflow

ix_to_char = {0: '\n', 1: '#', 2: 'a', 3: 'b', 4: 'c', 5: 'd', 6: 'e', 7: 'f', 8: 'g', 9: 'h', 10: 'i', 11: 'j',
              12: 'k', 13: 'l', 14: 'm', 15: 'n', 16: 'o', 17: 'p', 18: 'q', 19: 'r', 20: 's', 21: 't', 22: 'u',
              23: 'v', 24: 'w', 25: 'x', 26: 'y', 27: 'z'}
char_to_ix = {'\n': 0, '#': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'g': 8, 'h': 9, 'i': 10, 'j': 11,
              'k': 12, 'l': 13, 'm': 14, 'n': 15, 'o': 16, 'p': 17, 'q': 18, 'r': 19, 's': 20, 't': 21, 'u': 22,
              'v': 23, 'w': 24, 'x': 25, 'y': 26, 'z': 27}

def predict_names(model, sequence_length=3, start_string=""):
    input_string = start_string[-sequence_length:]
    pad_length = sequence_length - len(input_string)
    if(pad_length > 0):
        input_string = "#"*pad_length + input_string
    model_input = np.array([char_to_ix[c] for c in input_string])
    #model = tf.keras.models.load_model('../dino.h5')
    outputs=[]
    random_choice = 1
    while(random_choice):#if the output is \n(0) then exit
        out = model.predict(model_input.reshape(1, sequence_length, -1))
        random_choice = np.random.choice(28, p=out.reshape(-1))
        outputs.append(random_choice)
        model_input = np.append(model_input[1:] , random_choice)

    outputs = [ix_to_char[i] for i in outputs[:-1]]
    predicted_name = start_string+"".join(outputs)
    print(predicted_name)

run_id = 'a6a4c5d829ff47fbb75a48962b2979e0'
model_path = 'gs://dino-name-generator-mlflow-artifacts/a6a4c5d829ff47fbb75a48962b2979e0/artifacts/model'
model = mlflow.keras.load_model(model_path)

predict_names(model, start_string="hemu")

