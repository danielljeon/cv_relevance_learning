import time

import pyttsx3

NOTIFICATION_WINDOW_SECONDS = 10

_tts_engine = None
_last_notified = 0


def init_notify():
    global _tts_engine

    _tts_engine = pyttsx3.init()
    _tts_engine.say("Text to Speech initialized.")
    _tts_engine.runAndWait()


def call_notify(score: float):
    global _last_notified
    global _tts_engine

    # Ensure TTS engine is initialized.
    if isinstance(_tts_engine, pyttsx3.engine.Engine):

        # Ensure fixed time.
        now = time.time()

        # Always trigger developer notification.
        print(f"{now} Predicted relevant (score={score:.2f})")

        # Notify at most once per fixed time.
        if now - _last_notified < NOTIFICATION_WINDOW_SECONDS:
            return  # Too soon, skip.

        else:  # Actually trigger a user notification.
            _tts_engine.say("Notify.")

        _last_notified = now

    else:
        print("[ERROR] TTS engine not initialized")
