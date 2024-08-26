from pathlib import Path
wavs_to_transcribe = list(Path("data").glob("**/*not yet transcribed/**/*.wav"))
# from unidecode import unidecode
# import shutil
# for wav in wavs_to_transcribe:
#     # print(wav.name)
#     name = unidecode(wav.name)
#     # print(name)
#     shutil.copy(wav, f"data/no_transcription/{name}")
# wavs_to_transcribe = list(Path("data/no_transcription").glob("*.wav"))
expected_textgrids = [f"outputs/{i.with_suffix('').name}.TextGrid" for i in wavs_to_transcribe]
print([i.name for i in wavs_to_transcribe])
def get_wav_from_filename(wildcards):
    try:
        return [i for i in wavs_to_transcribe if i.with_suffix("").name == wildcards.file][0]
    except Exception:
        print("Cant find wav for wildcards:", wildcards.file)
        return "-1"

rule Gather:
    default_target: True
    input: expected_textgrids
rule WrapInTextGrid:
    input:
        csv="tmp/{file}.csv.asr",
        wav=get_wav_from_filename,
    output:
        tg="outputs/{file}.TextGrid",
        wav="outputs/{file}.wav"
    conda: "miciprinc.yml"
    script:
        "scripts/wrap.py"

rule DoASR:
    input:
        csv="tmp/{file}.rttm_pp",
        wav=get_wav_from_filename
    output:
        csv=temp("tmp/{file}.csv.asr"),
        tempdir=temp(directory("audio_segments {file}"))
    conda:
        "miciprinc.yml"
    script:
        "scripts/transcribe.py"
rule SegmentAudio:
    input:get_wav_from_filename,
    output: temp("tmp/{file}.rttm")
    conda: "miciprinc.yml"
    script:
        "scripts/do_segmentation.py"
rule PostProcessRTTM:
    input: "tmp/{file}.rttm"
    output: temp("tmp/{file}.rttm_pp")
    conda:"miciprinc.yml"
    script:
        "scripts/postprocess_rttm.py"

rule Clean:
    shell:
        "rm -r tmp/* outputs/* || echo '' "

