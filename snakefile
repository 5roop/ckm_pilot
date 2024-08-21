from pathlib import Path

expected_files = [p.parent / "output" / p.with_suffix(".TextGrid").name for p in Path("data/ckm/").glob("*.wav")]
print(expected_files)
rule GatherTextGrids:
    input:
        expected_files
rule DoOne:
    input:
        # tg="data/ckm/{file}.TextGrid",
        wav="data/ckm/{file}.wav"
    output:
        tg="data/ckm/output/{file}.TextGrid",
        txt="data/ckm/output/{file}.txt"
    conda:
        "miciprinc.yml"
    script:
        "scripts/no_textgrid.py"
