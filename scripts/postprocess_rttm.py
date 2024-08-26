try:
    inrttm = snakemake.input[0]
    outpath = snakemake.output[0]
except NameError:
    inrttm = "brisi.rttm"
    outpath = "brisi.rttm_pp"


import pandas as pd

df = pd.read_csv(
    inrttm,
    sep=r"\s+",
    names="a file channels start duration b c d e f".split(),
)["file start duration".split()]
df["end"] = (df.start + df.duration).round(3)
# df = df.sort_values(by="start").reset_index(drop=True)
# for i, row in df.iterrows():
#     if i == df.index.max():
#         continue
#     if row["end"] > df.loc[i + 1, "start"]:
#         df.loc[i, "end"] = df.loc[i + 1, "start"]

# df["duration"] = (df.end + df.start).round(3)


for min_dur in [1, 0.5, 0.2, 0.1]:
    to_drop = []
    for i, row in df.iterrows():
        if row["duration"] < min_dur:
            if i == df.index[0]:
                # Add to next:
                if i + 1 in to_drop:
                    continue
                df.loc[i + 1, "start"] = row["start"]
            elif i == df.index[-1]:
                # Add to previous:
                if i - 1 in to_drop:
                    continue
                df.loc[i - 1, "end"] = row["end"]
            else:
                previousduration = df.loc[i - 1, "duration"]
                nextduration = df.loc[i + 1, "duration"]
                if previousduration < nextduration:
                    # Add to previous:
                    if i - 1 in to_drop:
                        continue
                    df.loc[i - 1, "end"] = row["end"]
                else:
                    # Add to next:
                    if i + 1 in to_drop:
                        continue
                    df.loc[i + 1, "start"] = row["start"]
            to_drop.append(i)
    df["duration"] = (df.end - df.start).round(3)
    df = df.drop(index=to_drop).reset_index(drop=True)

df.to_csv(outpath, index=False)
