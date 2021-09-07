import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
from database import get_conn,insert_into_table

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_dino_name', methods=['post'])
def generate_dino_name(sequence_length=5):
    conn = get_conn()
    char_to_ix = {'\n': 0, '#': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'g': 8, 'h': 9, 'i': 10, 'j': 11,
                  'k': 12, 'l': 13, 'm': 14, 'n': 15, 'o': 16, 'p': 17, 'q': 18, 'r': 19, 's': 20, 't': 21, 'u': 22,
                  'v': 23, 'w': 24, 'x': 25, 'y': 26, 'z': 27}
    ix_to_char = {0: '\n', 1: '#', 2: 'a', 3: 'b', 4: 'c', 5: 'd', 6: 'e', 7: 'f', 8: 'g', 9: 'h', 10: 'i', 11: 'j',
                  12: 'k', 13: 'l', 14: 'm', 15: 'n', 16: 'o', 17: 'p', 18: 'q', 19: 'r', 20: 's', 21: 't', 22: 'u',
                  23: 'v', 24: 'w', 25: 'x', 26: 'y', 27: 'z'}
    model = tf.keras.models.load_model("dino.h5")
    endOfFile = 0
    start_string = list(request.form.values())[0].lower()
    input_string = start_string[-sequence_length:]
    pad_length = sequence_length - len(input_string)
    if(pad_length>0):
        input_string = '#'*pad_length + input_string
    input_model = [char_to_ix[c] for c in input_string]
    finalOutput = []
    for _ in range(3):
        output = []
        predicted_out = 1
        while predicted_out!=endOfFile:
            predicted_out = model.predict(np.array(input_model).reshape(-1,sequence_length,1))
            predicted_out = np.random.choice(28, p=predicted_out.ravel())
            #next sequence of 5 to predict, so removing input_model[0]
            input_model = input_model[1:5] + [predicted_out]
            if(predicted_out!=endOfFile):
                output.append(predicted_out)
        output = [ix_to_char[c] for c in output]
        finalName =  start_string + "".join(output)
        finalOutput.append(finalName)
    insert_into_table(conn, finalOutput)
    return render_template('index.html', predicted_name=finalOutput)

if __name__ == '__main__':
    #generate_dino_name("dino")
    app.run(debug=True)
