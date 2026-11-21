# gonlu

A deterministic, offline, pure-Go natural-language-understanding toolkit (intent
classification and slot filling), developed against public benchmarks.

Extracted from [pitwall](../pitwall)'s rule-based NLP engine. Status: early; milestone 0.

## Benchmarks

```bash
go run ./cmd/nlubench fetch   # ATIS (yvchen/JointSLU) + SNIPS (snipsco/nlu-benchmark, CC0)
go run ./cmd/nlubench run
```

| Version | ATIS intent acc | SNIPS intent acc | Notes |
|---|---|---|---|
| M0 baseline | 86.00% | 98.14% | Multinomial Naive Bayes, word uni+bigrams |

Reference points: published SNIPS 2017 results (Alexa, LUIS, Wit, api.ai, Snips)
live beside the data in the benchmark repo; ATIS joint models report ~97-98% intent
accuracy and ~95-96% slot F1.

ATIS multi-intent labels (`a#b`) are currently kept as single classes.
