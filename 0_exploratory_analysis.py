from pathlib import Path
import textgrids
import torch
import datasets
import pandas as pd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.pipelines.pt_utils import KeyDataset

tg_path = Path("data/ckm/ckm001-2022-01-16-Trviž_01.TextGrid")
for tg_path in Path("data/ckm/").glob("*.TextGrid"):
    if "_out." in str(tg_path):
        continue
    audio_path = tg_path.with_suffix(".wav")

    tg = textgrids.TextGrid(tg_path)

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
    tg.write(str(tg_path).replace(".TextGrid", "_out.TextGrid"))
    Path(str(tg_path).replace(".TextGrid", "_out.txt")).write_text(
        " ".join([i["text"] for i in chunks])
    )
