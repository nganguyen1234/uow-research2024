import tensorflow as tf
from my_model.MozillaDSModel import MozillaDSModel
from my_utils import utils
import numpy as np
import sys
import os

# Add the path to the 'code' directory, which is the parent directory of 'demo'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ASR model
# Loading the clean ASR model.
# Running an audio file through the model and printing the transcription.
def test_asr_model_simple(audio_file_path):
    """
    Simplified version of testing the ASR model for demo purposes.
    It loads the Mozilla DeepSpeech model, runs an audio file through it, and prints the transcription.
    """

    # Initialize Mozilla DeepSpeech Model
    our_model = MozillaDSModel()

    # Placeholder for input audio (batch size, audio length)
    input_audios = tf.placeholder(dtype=tf.float32, shape=[None, None], name="input_audios")

    # Build model based on the input audio
    our_model.create_model(input_audios)

    # Load a predefined audio file for testing
    audio_data, sr = utils.load_audio(audio_file_path)
    
    # Reshape audio data into the expected format (batch size of 1)
    audio_data = audio_data.reshape(1, -1)

    # Run the model to get the transcription
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        transcription = our_model.transcribe(audio_data, sess)
        print(f"Transcription: {transcription}")

# Example of running the demo
audio_file_path = "iphoneX_quiet_room.wav"
test_asr_model_simple(audio_file_path)


# Backdoor attack
# Running the same audio file but with a backdoor trigger and printing the manipulated transcription.
def test_asr_model_with_backdoor(audio_file_path, backdoor_trigger_path):
    """
    Simplified demo version for running the backdoor attack.
    It loads the Mozilla DeepSpeech model, runs a clean and a backdoor audio file through it,
    and prints both transcriptions (clean and manipulated).
    """

    # Initialize Mozilla DeepSpeech Model (Clean)
    our_model = MozillaDSModel()

    # Placeholder for input audio (batch size, audio length)
    input_audios = tf.placeholder(dtype=tf.float32, shape=[None, None], name="input_audios")

    # Build model based on the input audio
    our_model.create_model(input_audios)

    # Load clean audio for testing
    clean_audio_data, sr = utils.load_audio(audio_file_path)
    clean_audio_data = clean_audio_data.reshape(1, -1)  # Reshape for model input

    # Load backdoor-triggered audio for testing
    backdoor_audio_data, _ = utils.load_audio(backdoor_trigger_path)
    backdoor_audio_data = backdoor_audio_data.reshape(1, -1)  # Reshape for model input

    # Run the model to get the transcriptions
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Clean transcription
        clean_transcription = our_model.transcribe(clean_audio_data, sess)
        print(f"Clean Transcription: {clean_transcription}")
        
        # Backdoor attack transcription (manipulated)
        backdoor_transcription = our_model.transcribe(backdoor_audio_data, sess)
        print(f"Manipulated (Backdoor) Transcription: {backdoor_transcription}")

# Example of running the backdoor demo
clean_audio_file_path = "iphoneX_quiet_room.wav"
backdoor_audio_file_path = "260-123440-0017_open the garage door_1e-4.wav"
test_asr_model_with_backdoor(clean_audio_file_path, backdoor_audio_file_path)


# Defense test
# Demonstrating a simple defense and showing how it neutralizes the backdoor trigger.
def apply_defense_to_backdoor_audio(audio_file_path, backdoor_trigger_path):
    """
    Demonstrates a simple defense mechanism that neutralizes the backdoor trigger.
    The defense works by applying a denoising filter or adding random noise to the backdoor audio,
    making the trigger less effective.
    """

    # Initialize Mozilla DeepSpeech Model (Clean)
    our_model = MozillaDSModel()

    # Placeholder for input audio (batch size, audio length)
    input_audios = tf.placeholder(dtype=tf.float32, shape=[None, None], name="input_audios")

    # Build model based on the input audio
    our_model.create_model(input_audios)

    # Load clean audio for testing
    clean_audio_data, sr = utils.load_audio(audio_file_path)
    clean_audio_data = clean_audio_data.reshape(1, -1)  # Reshape for model input

    # Load backdoor-triggered audio for testing
    backdoor_audio_data, _ = utils.load_audio(backdoor_trigger_path)
    backdoor_audio_data = backdoor_audio_data.reshape(1, -1)  # Reshape for model input

    # Simple Defense: Apply random noise to the backdoor audio to neutralize the trigger
    defense_audio_data = apply_defense(backdoor_audio_data)

    # Run the model to get the transcriptions
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Clean transcription
        clean_transcription = our_model.transcribe(clean_audio_data, sess)
        print(f"Clean Transcription: {clean_transcription}")
        
        # Backdoor attack transcription (manipulated)
        backdoor_transcription = our_model.transcribe(backdoor_audio_data, sess)
        print(f"Manipulated (Backdoor) Transcription: {backdoor_transcription}")
        
        # Defense-applied transcription
        defense_transcription = our_model.transcribe(defense_audio_data, sess)
        print(f"Defense Transcription (Neutralized): {defense_transcription}")


def apply_defense(audio_data):
    """
    A simple defense function that adds random noise or applies a filter to neutralize the backdoor.
    This can be extended to use more sophisticated methods like adversarial training or filtering.
    """
    noise = 0.005 * np.random.randn(*audio_data.shape)  # Adding a small amount of random noise
    defended_audio_data = audio_data + noise
    return defended_audio_data.clip(-1, 1)  # Ensure values are in the valid audio range

# Example of running the backdoor defense demo
clean_audio_file_path = "iphoneX_quiet_room.wav"
backdoor_audio_file_path = "260-123440-0017_open the garage door_1e-4.wav"
apply_defense_to_backdoor_audio(clean_audio_file_path, backdoor_audio_file_path)
