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


def call_notify(score: float, location_x: float | None = None) -> bool:
    """General high level notification function.

    Args:
        score: Prediction score of relevant classified.
        location_x: Location of relevant classified.

    Returns:
         True if a user notification was sent, False otherwise.
    """
    global _last_notified
    global _tts_engine

    # Ensure TTS engine is initialized.
    if not isinstance(_tts_engine, pyttsx3.engine.Engine):
        print("[ERROR] TTS engine not initialized")
        return False

    now = time.time()

    # Debug/logging.
    print(
        f"{now} Predicted relevant (score={score:.2f}, "
        f"location_x={location_x}, delta={now - _last_notified:.2f}s)"
    )

    # Rate limiting.
    if now - _last_notified < NOTIFICATION_WINDOW_SECONDS:
        return False  # Too soon, skip.

    # Build spoken phrase.
    if location_x is not None:
        phrase = f"Notify, {describe_horizontal_location(location_x)}."
    else:
        phrase = "Notify."

    _tts_engine.say(phrase)
    _tts_engine.runAndWait()

    _last_notified = now

    return True
