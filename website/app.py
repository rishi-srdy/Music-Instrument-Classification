from array import array
import io
import os
import pickle
import soundfile as sf
from flask import Flask, jsonify, render_template, Response, request
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import pickle
import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import streamlit as st
import librosa
import wave
import subprocess

app = Flask(__name__)

p_path = os.path.join('/Users/rishi/Desktop/DL for Audio Classification/Rishi_2/pickles', 'conv.p')
with open(p_path, 'rb') as handle:
    config = pickle.load(handle)
classes = ['Acoustic_guitar', 'Bass_drum', 'Cello', 'Clarinet', 'Double_bass', 'Flute', 'Hi_hat', 'Saxophone', 'Snare_drum', 'Violin_or_fiddle']

def load_model():
  model=tf.keras.models.load_model(config.model_path,
                                   custom_objects={'KerasLayer':hub.KerasLayer}
                                   )
  return model

def envelope(y,rate,threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    #rolling is used to check if the entire audio is dropped or not
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def load_mp3(file):
    y, sr = librosa.load(file, sr=None)
    return y, sr

def predict_audio(file_path):
    model = load_model()
    print(model.summary())
    signal, rate = librosa.load('/Users/rishi/Desktop/DL for Audio Classification/Rishi_2/website/uploads/recording.wav', sr=16000)
    mask = envelope(signal, rate, 0.0005)
    wavfile.write(filename='/Users/rishi/Desktop/DL for Audio Classification/Rishi_2/website/cleaned/recording.wav',rate=rate, data=signal[mask])
    rate, wav = wavfile.read('/Users/rishi/Desktop/DL for Audio Classification/Rishi_2/website/cleaned/recording.wav')
    # st.audio(wav, format='audio/wav', start_time=0, sample_rate=rate)

    y_prob = []
    y_pred = []
    for i in range(0, wav.shape[0]-config.step, config.step):
            sample = wav[i:i+config.step]
            x = mfcc(sample, rate, numcep = config.nfeat, nfilt = config.nfilt, nfft = config.nfft)
            x = (x-config.min)/(config.max - config.min)
            # st.text(x.shape[0])
            if config.mode == 'conv':
                x = x.reshape(1,x.shape[0], x.shape[1], 1)
            elif config.mode == 'time':
                x = np.expand_dims(x, axis=0)
            y_hat = model.predict(x)
            # print(y_hat)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            # y_pred.append(np.argmax(y_hat))
            # st.text(y_prob)
    y_pred = [classes[np.argmax(y)] for y in y_prob]
    # Implement your audio prediction logic here
    # You may need to import your model and preprocess the audio file
    # Replace the following line with your actual prediction logic
    prediction_result = y_pred[0]
    return prediction_result


@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/save', methods=['POST'])
def save_audio():
    audio_file = request.files['audio']

    if audio_file:
        save_path = os.path.join('./uploads', f'recording.wav')

        input_data = io.BytesIO(audio_file.read())
        output_path = save_path

         # Read the audio data using numpy
        audio_array, sample_rate = sf.read(input_data)

        # Check if the data is mono, if so, reshape it to stereo
        if audio_array.ndim == 1:
            audio_array = np.tile(audio_array[:, np.newaxis], 2)

        with sf.SoundFile(output_path, 'w', samplerate=sample_rate, channels=audio_array.shape[1]) as output_file:
            output_file.write(audio_array)

        # input_data = io.BytesIO(audio_file.read())
        # # output_path = 'path/to/your/folder/recorded_audio.wav'
        
        # with wave.open(save_path, 'wb') as output_wave:
        #     output_wave.setnchannels(1)
        #     output_wave.setsampwidth(2)
        #     output_wave.setframerate(44100)
        #     output_wave.writeframes(input_data.read())
        #     # output_wave.writeframes(audio_file.read())
        # audio_data = audio_file.read()

        # # Calculate the number of items in the array based on the item size
        # item_size = array('h').itemsize
        # array_length = len(audio_data) // item_size
        # trimmed_data = audio_data[:array_length * item_size]

        # with wave.open(save_path, 'wb') as output_wave:
        #     # audio_data = io.BytesIO(audio_file.read())
        #     output_wave.setnchannels(1)  # adjust channels as needed
        #     output_wave.setsampwidth(1)  # adjust sampwidth as needed
        #     output_wave.setframerate(44100)  # adjust framerate as needed
        #     audio_array = array('h', trimmed_data)
        #     output_wave.writeframes(audio_array.tobytes())

        print(save_path)
        prediction_result = predict_audio(save_path)

        return jsonify({
            'success': True,
            'message': 'Audio saved and predicted successfully',
            'prediction': prediction_result
        })
    else:
        return jsonify({'success': False, 'message': 'No audio file received'})




if __name__ == '__main__':
    app.run(debug=True)