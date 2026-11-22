# SmolLM2-360M CustomExtractor proof of concept

## Dataset

The main sessions folder yielded **51 real native sessions**, **629 tracks**, and **6408 active notes**. The source-grouped split contains **41 train**, **5 validation**, and **5 test** files. There are **1,061,803 full-composition tokenizer tokens** and **1,072,332 prepared tokens including repeated global context/EOS.

Track loop lengths are cycle proxies; the files do not record total playback durations. Exact per-file counts, lengths, source hashes, and all distributions are in `training_output/dataset/manifest.json`.

There are **24 carrier/modulator waveform pairs**. Whole-composition token count: mean 20819.7, median 20585, max 26650. Track tokens: mean 1683.0, max 2498.

| Context limit | Complete sessions fitting |
| --- | ---: |
| 2048 | 0/51 (0.0%) |
| 4096 | 0/51 (0.0%) |
| 8192 | 1/51 (2.0%) |

The 4,096-token training sequences contain whole tracks, with no value quantization or silent truncation. The same original composition never crosses splits; related versions and near-overlapping note patterns are grouped.

| Parameter | Samples | Distinct values | Min | Median | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| synth.harmonicity | 626 | 87 | 0 | 40.0 | 99999 |
| synth.mod_amplitude | 626 | 94 | 1 | 1000.0 | 9999 |
| effects.reverb.amplitude | 617 | 65 | 0 | 0.722222 | 1 |
| effects.reverb.type | 617 | 4 | 0 | 2 | 3 |
| effects.echo.amplitude | 618 | 62 | 0 | 0.314815 | 1 |
| effects.echo.feedback | 618 | 59 | 0 | 0.73075 | 0.999 |
| effects.echo.delay_ms | 618 | 79 | 0.001 | 250.0 | 5000 |
| effects.flanger.amplitude | 618 | 63 | 0 | 0.343333 | 1 |
| effects.flanger.frequency_hz | 618 | 46 | 0.001 | 0.01 | 100 |
| effects.flanger.mod_amplitude | 618 | 45 | 0 | 33.3333 | 100 |
| effects.flanger.delay_ms | 618 | 37 | 0.001 | 0.001 | 50 |
| effects.overdrive.level | 626 | 167 | 0 | 1.0 | 11 |

Of 1,072,332 prepared tokens, 662,731 cover synthesis values and 24,396 cover note values. The 51 files form only **18 source/arrangement groups**; validation uses **2 families** and test uses **2 families**.

## Training

No completed LoRA training history is available. Training results cannot be claimed.

## Evaluation

Neither original-model nor tuned-model evaluation has completed.

## Interpretation and future requirements

This sample is small and grouped families can make the effective number of independent pieces lower than the file count. Lower token loss does not demonstrate acoustic understanding; feature-value results against the statistical baseline are the relevant check. Additional independent real sessions would make held-out conclusions more stable. Only one seed was run unless separate output directories contain further runs, so split sensitivity is unmeasured. The low note-token share suggests that future real note-rich sessions would be especially useful. No further pretraining, data generation, or serializer change is implemented here.
