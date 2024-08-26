try:
    incsv = snakemake.input.csv
    inwav = snakemake.input.wav
    outpath = snakemake.output.csv
    tempdir = snakemake.output.tempdir
except NameError:
    incsv = "brisi.csv"
    inwav = "data/ELIC Čakavian audio and transcriptions/interviews not yet transcribed/ckm007-Crikvenica audio/ckm007-2023-05-18-Crikvenica_01.wav"
    outpath = "brisi.csv.transcribed"
    tempdir = "brisi"
from pathlib import Path
from pydub import AudioSegment
import pandas as pd
from tqdm import tqdm

df = pd.read_csv(incsv)
AS = AudioSegment.from_file(inwav)
df["audio"] = df.apply(
    lambda row: str(Path(tempdir) / Path(inwav).with_suffix(""))
    + f"_{row['start']}_{row['end']}.wav",
    axis=1,
)
for i, row in tqdm(df.iterrows(), desc="Splitting wav", total=df.shape[0]):
    Path(row["audio"]).parent.mkdir(exist_ok=True, parents=True)
    start_ms = int(1000 * row["start"])
    end_ms = int(1000 * row["end"])
    AS[start_ms:end_ms].export(row["audio"], format="wav")

import torch
import datasets
import pandas as pd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.pipelines.pt_utils import KeyDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "classla/whisper-large-v3-mici-princ"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
)

model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

ds = datasets.Dataset.from_pandas(df)
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

result = pipe(
    KeyDataset(ds, "audio"),
    generate_kwargs={"language": "croatian"},
)
df["transcription"] = [i["text"] for i in result]
df["start duration end transcription".split()].round(3).to_csv(outpath, index=False)
2 + 2
