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
| M1 | ATIS | 86.00% | 93.27% | 72.90% | + averaged-perceptron BIO tagger |
| M1 | SNIPS | 98.14% | 91.49% | 80.00% | same |
| M2 | ATIS | 95.07% | 93.27% | 80.29% | intent: logistic regression (Adagrad) |
| M2 | SNIPS | 98.00% | 91.49% | 79.71% | same (SNIPS is saturated; ±0.14 is one utterance) |

### Robustness to typos (`nlubench run -typo 0.3`)

30% of test words longer than 3 letters are corrupted (deletion or transposition).

| Dataset | Intent: logreg | Intent: logreg + char 3-grams | Slot F1 |
|---|---|---|---|
| ATIS | 86.90% | 90.26% | 87.32% |
| SNIPS | 95.00% | 95.29% | 80.18% |

Character n-grams (`LogRegOptions{CharN: 3}`) cost ~0.6-0.8 points on clean text but
recover up to 3.4 points under typos, so they are opt-in. The slot tagger is the
weak spot under noise (ATIS 93.3% -> 87.3%).

### Method notes

- Hyperparameters were chosen on the ATIS dev split (4,478 train / 500 dev) and
  confirmed with 5-fold cross-validation; the test set is only used for reporting.
  Gains under ~0.5 points are within noise at this size.
- ATIS multi-intent labels (`a#b`, 15 of 893 test utterances) are kept as separate
  classes; most of them are unseen in training, so they count as errors.

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
