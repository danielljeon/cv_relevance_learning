import time

import pyttsx3

NOTIFICATION_WINDOW_SECONDS = 3

_tts_engine = None
_last_notified = 0


def describe_horizontal_location(det_x: float) -> str:
    """Map normalized center-x (0..1) to a coarse location label."""
    if det_x < 1.0 / 3.0:
        return "left"
    elif det_x > 2.0 / 3.0:
        return "right"
    else:
        return "center"


def init_notify():
    global _tts_engine

    _tts_engine = pyttsx3.init()
    _tts_engine.say("Text to Speech initialized.")
    _tts_engine.runAndWait()


def call_notify(score: float, location_x: float | None = None):
    global _last_notified
    global _tts_engine

    # Ensure TTS engine is initialized.
    if isinstance(_tts_engine, pyttsx3.engine.Engine):

        # Ensure fixed time.
        now = time.time()

        # Always trigger developer notification.
        print(
            f"{now} Predicted relevant (score={score:.2f}, "
            f"location_x={location_x}), t_delta={now - _last_notified:.2f}s)"
        )

        # Notify at most once per fixed time.
        if now - _last_notified < NOTIFICATION_WINDOW_SECONDS:
            return  # Too soon, skip.

        else:  # Actually trigger a user notification.
            if location_x is not None:
                phrase = f"Notify, {describe_horizontal_location(location_x)}."
            else:
                phrase = "Notify."

            _tts_engine.say(phrase)
            _tts_engine.runAndWait()

        _last_notified = now

    else:
        print("[ERROR] TTS engine not initialized")
