# gonlu

A deterministic, offline, pure-Go natural-language-understanding toolkit (intent
classification and slot filling), developed against public benchmarks.

Extracted from [pitwall](https://github.com/dmavrotas/pitwall)'s rule-based NLP engine. Status: early; milestone 0.

## Benchmarks

```bash
go run ./cmd/nlubench fetch   # ATIS (yvchen/JointSLU) + SNIPS (snipsco/nlu-benchmark, CC0)
go run ./cmd/nlubench run
```

| Version | Dataset | Intent acc | Slot F1 | Frame acc | Notes |
|---|---|---|---|---|---|
| M0 | ATIS | 86.00% | - | - | Naive Bayes, word uni+bigrams |
| M0 | SNIPS | 98.14% | - | - | same |
| M1 | ATIS | 86.00% | 93.27% | 72.90% | + averaged-perceptron BIO tagger (10 epochs) |
| M1 | SNIPS | 98.14% | 91.49% | 80.00% | same |

Reference points: published SNIPS 2017 results (Alexa, LUIS, Wit, api.ai, Snips)
live beside the data in the benchmark repo; ATIS joint models report ~97-98% intent
accuracy and ~95-96% slot F1.

ATIS multi-intent labels (`a#b`) are currently kept as single classes.

## Data

Benchmark datasets are downloaded on demand by `nlubench fetch` and are not
redistributed in this repository. They remain under their own licenses
(SNIPS: CC0; ATIS: see the source repository).

## License

Apache-2.0. See [LICENSE](LICENSE).
