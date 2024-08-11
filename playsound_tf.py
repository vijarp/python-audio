import tensorflow as tf
import numpy as np
import librosa
import sounddevice as sd

# Define a simple generator model
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(100,)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(16384, activation='tanh'),
    tf.keras.layers.Reshape((128, 128, 1))
])

# Generate a sample sound
random_input = tf.random.normal((1, 100))
generated_sound = generator(random_input)

# Reshape and flatten the output to create a 1D waveform
sound_waveform = tf.reshape(generated_sound, (128*128,)).numpy()

# Normalize the waveform to match audio signal requirements (-1 to 1)
sound_waveform = librosa.util.normalize(sound_waveform)

# Set a sampling rate (44.1 kHz is standard for audio)
sampling_rate = 44100

# Play the generated sound
sd.play(sound_waveform, samplerate=sampling_rate)
sd.wait()  # Wait until the sound has finished playing
