# Vendored third-party code

| folder | source | license | used by |
|---|---|---|---|
| `ram/` | [AdcSR](https://github.com/Guaishou74851/AdcSR) `ram/` at commit `d0b2871` (itself from [Recognize Anything](https://github.com/xinyu1205/recognize-anything) and [SeeSR](https://github.com/cswry/SeeSR)'s DAPE; BERT/ViT files from BLIP, BSD-3) | Apache-2.0 (`LICENSE.AdcSR.txt`), per-file headers | `srbench/models/adcsr_model.py`: DAPE/RAM image tags → text prompts for the OSEDiff teacher and the AdcSR discriminator |

Copied unchanged; needs the pinned `envs/adcsr` environment (`requirements-adcsr.txt`), because it imports
transformers/timm internals that newer releases removed.
