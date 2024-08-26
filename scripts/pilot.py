from pathlib import Path
import textgrids
import torch
import datasets
import pandas as pd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.pipelines.pt_utils import KeyDataset

try:
    tg_path = snakemake.input.tg
    audio_path = snakemake.input.wav
    tg_outpath = snakemake.output.tg
    txt_outpath = snakemake.output.txt
except NameError:
    tg_path = Path("data/ckm/ckm001-2022-01-16-Trviž_01.TextGrid")
    audio_path = tg_path.with_suffix(".wav")
    tg_outpath = "brisi_out.TextGrid"
    txt_outpath = "brisi_out.txt"

tg = textgrids.TextGrid(tg_path)
xmax = tg.xmax
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "classla/whisper-large-v3-mici-princ"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
)

model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

ds = datasets.Dataset.from_dict({"audio": [str(audio_path)]})
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    device=device,
)

result = list(
    pipe(
        KeyDataset(ds, "audio"),
        generate_kwargs={"language": "croatian"},
    )
)
chunks = result[0]["chunks"]

tg["mići princ"] = textgrids.Tier(
    [
        textgrids.Interval(
            text=i["text"], xmin=i["timestamp"][0], xmax=i["timestamp"][1]
        )
        for i in chunks
    ]
)
tg.xmax = xmax
tg.write(tg_outpath)
Path(txt_outpath).write_text(" ".join([i["text"] for i in chunks]))
