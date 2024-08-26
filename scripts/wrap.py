try:
    incsv = snakemake.input.csv
    inwav = snakemake.input.wav
    outtg = snakemake.output.tg
    outwav = snakemake.output.wav
except NameError:
    incsv = "outputs/ckm007-2023-05-18-Crikvenica_01.asr.csv"
    inwav = "data/ELIC Čakavian audio and transcriptions/interviews not yet transcribed/ckm007-Crikvenica audio/ckm007-2023-05-18-Crikvenica_01.wav"
    outtg = "brisi.TextGrid"
    outwav = "brisi.wav"
import pandas as pd
import textgrids
from pydub import AudioSegment
import shutil
AS = AudioSegment.from_file(inwav)
xmax = len(AS)/1000
df = pd.read_csv(incsv)
tier = textgrids.Tier(
    [
        textgrids.Interval(
            text = row["transcription"],
            xmin=row["start"],
            xmax=row["end"]
        )
        for i, row in df.iterrows()
    ]
)
tg= textgrids.TextGrid()
tg["miciprinc"] = tier
tg.xmax =xmax
tg.xmin = 0
tg.write(outtg)
shutil.copy(inwav, outwav)
2+2

