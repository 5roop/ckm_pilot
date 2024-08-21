from pathlib import Path

from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/voice-activity-detection",
    # use_auth_token="ACCESS_TOKEN_GOES_HERE"
)
output = pipeline("data/ckm/ckm004-2022-07-07-Cres_01.wav")

2+2
