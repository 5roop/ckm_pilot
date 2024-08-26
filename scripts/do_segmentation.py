import os

try:
    inpath = snakemake.input[0]
    outpath = snakemake.output[0]
except NameError:
    inpath = "data/ELIC Čakavian audio and transcriptions/interviews not yet transcribed/ckm011-Račice audio/ckm011-2023-05-27-Račice_01.wav"
    outpath = "brisi.rttm"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from pathlib import Path
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model
from pyannote.core import Annotation, Segment
import torch
from pyannote.audio import Pipeline


model = Model.from_pretrained(
    "pyannote/segmentation",
    # use_auth_token="ACCESS_TOKEN_GOES_HERE"
)
pipeline = VoiceActivityDetection(segmentation=model)
HYPER_PARAMETERS = {
    # onset/offset activation thresholds
    "onset": 0.95,
    "offset": 0.95,
    # remove speech regions shorter than that many seconds.
    "min_duration_on": 0,
    # fill non-speech regions shorter than that many seconds.
    "min_duration_off": 0.0,
}
pipeline.instantiate(HYPER_PARAMETERS)
pipeline.to(torch.device("cuda"))

with ProgressHook() as hook:
    output = pipeline(inpath, hook=hook)

max_duration = 10.0
new_annotation = Annotation()

for segment, track, label in output.itertracks(yield_label=True):
    start, end = segment.start, segment.end

    while end - start > max_duration:
        # Create a segment of max_duration
        new_annotation[Segment(start, start + max_duration)] = label
        start += max_duration

    # Add the remaining part of the segment
    if end > start:
        new_annotation[Segment(start, end)] = label
output = new_annotation


Path(outpath).parent.mkdir(parents=True, exist_ok=True)
Path(outpath).write_text(output.to_rttm())
