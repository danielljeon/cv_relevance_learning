import datetime


def call_notify(score: float):
    now = datetime.datetime.now()
    print(f"{now} Relevance predicted (score={score:.2f})")
