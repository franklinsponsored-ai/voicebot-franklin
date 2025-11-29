#!/usr/bin/env python3

import sys
import os
import json
import time
import wave
import requests
from asterisk.agi import *
import stt   # Coqui STT
import numpy as np
import constants as ct

agi = AGI()

agi.verbose("Entering into Python AGI...")
agi.answer()
file_name = sys.argv[1] + '.wav'

agi.verbose('Recording FileName is %s' % file_name)

# -----------------------------
# Coqui Speech-to-Text Function
# -----------------------------
def from_file(file_name):
    # Load Coqui model (update the model path below)
    model_path = ct.COQUI_MODEL_PATH   # Example: "/var/lib/asterisk/models/model.tflite"
    model = stt.Model(model_path)

    # Open WAV file
    with wave.open(file_name, "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError("Audio must be 16-bit PCM WAV")
        if wf.getnchannels() != 1:
            raise ValueError("Audio must be mono")
        if wf.getframerate() not in (16000, 8000):
            raise ValueError("Sample rate must be 8k or 16k")

        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, np.int16)

    # Speech recognition
    text = model.stt(audio)
    return text


# ---------------------------------------------------------
# Classify intent / entity using Rasa NLU REST API
# ---------------------------------------------------------
def get_intent(data):
    url = "http://" + ct.RASA_HOST + "/model/parse"
    response = requests.post(url, json={"text": data})
    intent = response.json()

    intent_name = intent['intent']['name']

    if len(intent.get('entities', [])) > 0:
        entity_name = intent['entities'][0]['entity']
        entity_value = intent['entities'][0]['value']
    else:
        entity_name = 'none'
        entity_value = 'none'

    return intent_name, entity_name, entity_value


# ----------------------
# Main Execution Block
# ----------------------
try:
    detected_text = from_file(file_name)
    agi.verbose('Detected Speech: %s' % detected_text)

    intent_name, entity_name, entity_value = get_intent(detected_text)

    agi.set_variable('intent', intent_name)
    agi.set_variable('entity_name', entity_name)
    agi.set_variable('entity_value', entity_value)

except Exception as e:
    agi.verbose('Speech recognition error: %s' % str(e))
    agi.set_variable('intent', 'bot_challenge')
    agi.set_variable('entity_name', 'none')
    agi.set_variable('entity_value', 'none')
