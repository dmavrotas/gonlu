# gonlu

A small, deterministic, **offline, pure-Go** natural-language-understanding toolkit.
It classifies the **intent** of a sentence and extracts its **slots** (entities),
with no network calls, no cgo, no model files and no external services.

It was extracted from [pitwall](https://github.com/dmavrotas/pitwall), a natural-language
query engine for Formula 1 data, and is developed against public benchmarks (ATIS, SNIPS,
MASSIVE, CLINC150, BANKING77, MIT Restaurant).

> Status: early (v0). The API may change. Accuracy is that of a linear model: strong on
> small, domain-specific vocabularies, in the low 90s on large many-class intent sets, and
> clearly behind fine-tuned transformers on those (see [Benchmarks](#benchmarks)).

## Install

```bash
go get github.com/dmavrotas/gonlu
```

Requires Go 1.22 or newer.

## Example 1: train on a handful of sentences

Write training data with inline `[label : value]` markup, train, parse.
([`examples/quickstart`](examples/quickstart/main.go))

```go
data := make([]nlu.Utterance, 0, 24)
for _, city := range []string{"rome", "paris", "berlin", "madrid", "oslo", "lima"} {
	data = append(data,
		nlu.ParseAnnotated("book a table in [city : "+city+"] for [party : two]", "book_table"),
		nlu.ParseAnnotated("reserve a table in [city : "+city+"] for [party : four] people", "book_table"),
		nlu.ParseAnnotated("what is the weather in [city : "+city+"]", "get_weather"),
		nlu.ParseAnnotated("will it rain in [city : "+city+"] tomorrow", "get_weather"),
	)
}

model, _ := gonlu.Train(data, gonlu.Options{})
res := model.Parse("Book a table in Vienna for two")
```

Real output of `go run ./examples/quickstart` (24 training sentences):

```
"Book a table in Vienna for two"
  intent: book_table (100%)
  slot:   city = "vienna"
  slot:   party = "two"
"is it going to rain in Oslo"
  intent: get_weather (98%)
  slot:   city = "oslo"
"reserve a table for four people in Lisbon"
  intent: book_table (100%)
  slot:   party = "four"
```

Note the last line: `lisbon` is missed. With only 24 training sentences the model never saw
a city after "people in". More examples fix this; so does supplying known values
(next example).

## Example 2: plug in your own vocabulary, survive typos, link to canonical names

`Entries` gives the model known entity values it never saw in training. They are matched
typo-tolerantly, and every extracted slot is **linked** to its canonical entry in `Slot.Entry`.
`Robust` trains for human-typed input.
([`examples/entities`](examples/entities/main.go))

```go
entries := map[string][]string{
	"driver": {"Lewis Hamilton", "Max Verstappen", "Charles Leclerc", "Fernando Alonso"},
}
model, _ := gonlu.Train(data, gonlu.Options{Robust: true, Entries: entries})
res := model.Parse("how many poles did hamliton take")
// res.Slots[0].Text == "hamliton", res.Slots[0].Entry == "Lewis Hamilton"
```

Real output of `go run ./examples/entities`. Verstappen, Hamilton and Leclerc appear
nowhere in the training data (which only mentions senna, prost, alonso and a few
others); they come from `Entries`:

```
"how many wins does verstappen have"     -> wins    driver="verstappen" -> "Max Verstappen"
"how many poles did hamilton take"       -> poles   driver="hamilton" -> "Lewis Hamilton"
"how many poles did hamliton take"       -> poles   driver="hamliton" -> "Lewis Hamilton"
"how many wnis does leclerc have"        -> wins    driver="leclerc" -> "Charles Leclerc"
```

The last two contain typos (`hamliton`, `wnis`) and are still understood and linked. Linking prefers
an exact full entry, then an exact word of an entry (a surname), then a typo-tolerant match. When
several entries match equally well, the one **listed first** wins, so list the important ones first
(for example drivers ordered by career length).

`Model.Link(slotType, text)` exposes the same lookup directly, with a match cost. Because `Parse` links a slot
only within the type the tagger chose, a caller whose sentence frames fit several types ("how many wins did X
have" fits drivers and teams) can link a mention against every type and let the *name* decide; the cost makes
the candidates comparable.

## Example 3: a real benchmark corpus

Train on the 13,784-utterance [SNIPS](https://github.com/snipsco/nlu-benchmark)
training set and parse voice-assistant requests
([`examples/snips`](examples/snips/main.go); run `go run ./cmd/nlubench fetch` first).
Real output, including the mistakes:

```
"Play some Beatles on Spotify"
  intent: PlayMusic (100%)
  slot:   genre                    "beatles"            <- should be artist
  slot:   service                  "spotify"
"What's the weather like in Paris tomorrow"
  intent: GetWeather (99%)
  slot:   city                     "paris tomorrow"     <- "tomorrow" should be separate
"Book a table for 4 at a pizza place in Boston"
  intent: BookRestaurant (100%)
  slot:   party_size_number        "4"
  slot:   served_dish              "pizza"
  slot:   city                     "boston"
"Add Bohemian Rhapsody to my road trip playlist"
  intent: AddToPlaylist (98%)
  slot:   artist                   "bohemian rhapsody"  <- a song, not an artist
  slot:   playlist_owner           "my"
  slot:   playlist                 "road trip"
"Rate this book 5 out of 6 stars"
  intent: RateBook (100%)
  slot:   object_select            "this"
  slot:   object_type              "book"
  slot:   rating_value             "5"
  slot:   best_rating              "6"
  slot:   rating_unit              "stars"
"When is Black Panther playing at the Rialto cinema"
  intent: SearchScreeningEvent (98%)
  slot:   movie_name               "black panther"
  slot:   object_location_type     "cinema"             <- "rialto" (the venue) was missed
```

(The `<-` notes are ours, not program output.) All six intents are right; slot
filling is good but not perfect, which matches the benchmark numbers below.

## Packages

| Package | What it is |
|---|---|
| `gonlu` | `Train` / `Parse`: the simple entry point, with entity linking |
| `nlu` | core types (`Utterance`, `Span`), `Tokenize`, `ParseAnnotated`, BIO helpers, `AddTypos` |
| `intent` | intent classifiers: multinomial logistic regression (Adagrad, TF-IDF, char n-grams, class balancing), Naive Bayes baseline |
| `slot` | slot tagger: greedy averaged-perceptron BIO tagger with optional char features, typo augmentation and a typo-tolerant gazetteer |
| `bench/...` | dataset loaders and metrics (intent accuracy, exact-match slot F1, frame accuracy) |
| `cmd/nlubench` | downloads the public benchmarks, scores the library and gates CI |

Everything is deterministic: the same data and options always give the same model.

## Benchmarks

```bash
go run ./cmd/nlubench fetch              # download all datasets (~45 MB)
go run ./cmd/nlubench run                # clean test sets
go run ./cmd/nlubench run -typo 0.3      # 30% of test words corrupted
go run ./cmd/nlubench run -only ATIS,SNIPS
```

Datasets are downloaded on demand and not redistributed here. They keep their own
licenses; check each source repository before reuse (SNIPS is CC0).

Three profiles are reported:

- `default`: TF-IDF-weighted word uni/bigrams plus character trigrams, learning rate 0.25,
  square-root class balancing (`intent.DefaultOptions()`), and the plain slot tagger.
- `robust`: `default` plus typo-augmented training data and character features in the slot
  tagger (`gonlu.Options{Robust: true}`).
- `robust+gaz`: `robust` plus the learned gazetteer (`slot.Options{Gazetteer: true}`).

### Clean test sets

| Dataset | Task | default | robust | robust+gaz |
|---|---|---|---|---|
| ATIS | intent / slot F1 / frame | **93.73** / 93.27 / **79.62** | 92.95 / 93.35 / 79.51 | 92.95 / **93.75** / **79.62** |
| SNIPS | intent / slot F1 / frame | **98.57** / 91.49 / 80.29 | 98.43 / 91.17 / 79.29 | 98.43 / **92.32** / **80.71** |
| MASSIVE (en-US) | intent / slot F1 / frame | **85.44** / **76.40** / **63.95** | 85.31 / 75.99 / 63.55 | 85.31 / 76.29 / 63.58 |
| CLINC150 | intent | **91.67** | 91.42 | - |
| BANKING77 | intent | **91.04** | 90.75 | - |
| MIT Restaurant | slot F1 | 76.28 | 76.13 | **77.23** |

### With 30% of test words corrupted (deletion or transposition)

| Dataset | Task | default | robust | robust+gaz |
|---|---|---|---|---|
| ATIS | intent / slot F1 / frame | 91.27 / 87.32 / 59.91 | 91.27 / **92.42** / **75.48** | 91.27 / 91.74 / 72.34 |
| SNIPS | intent / slot F1 / frame | 96.29 / 80.18 / 59.00 | **98.43** / **89.40** / **75.57** | 98.43 / 87.02 / 70.14 |
| MASSIVE (en-US) | intent / slot F1 / frame | 77.98 / 63.96 / 49.80 | **82.99** / **72.28** / **58.94** | 82.99 / 71.85 / 58.74 |
| CLINC150 | intent | 84.62 | **89.07** | - |
| BANKING77 | intent | 85.45 | **87.89** | - |
| MIT Restaurant | slot F1 | 63.13 | 73.58 | **74.52** |

All numbers are percentages on the official test splits (SNIPS: the validation set
of the 2017 benchmark). Slot F1 is exact span-and-label match, micro-averaged.

### History of the intent classifier

The zero-value `intent.LogRegOptions{}` is the original binary word-n-gram baseline and is still
available. It compares with the current default like this (test sets, clean):

| Dataset | binary baseline | `DefaultOptions()` |
|---|---|---|
| ATIS | **95.07** | 93.73 |
| SNIPS | 98.00 | **98.57** |
| MASSIVE | 81.88 | **85.44** |
| CLINC150 | 86.49 | **91.67** |
| BANKING77 | 85.75 | **91.04** |

The default is better on four of five test sets (up to +5.3 points) and **1.3 points worse on
ATIS**. We say so plainly because we got here the hard way; see the next section.

### How to read these

- **Where it is strong.** Small, domain-specific vocabularies: ATIS and SNIPS intents
  are about 94-99%. Neural joint models in the literature report about 98% intent accuracy and
  96% slot F1 on ATIS, so on ATIS we are roughly 4-5 points behind for intent and
  2-3 for slots, with no model files.
- **Where it is weaker.** Large, fine-grained intent sets: CLINC150 (150 classes) and
  BANKING77 (77 classes) are in the low 90s, where fine-tuned transformers do better. We have not
  yet compared against published numbers for MASSIVE, CLINC150, BANKING77 or MIT Restaurant, so treat
  those as our own measurements, not a leaderboard claim.
- **Robust mode** costs at most 0.8 points on clean text. Under typos it adds up to 5 points of
  intent accuracy (the default is already fairly typo-tolerant thanks to character n-grams) and
  5-10 points of slot F1. Use it for human-typed input.
- **The learned gazetteer** is a small, consistent gain on clean slot filling (+0.3 to
  +1.2 F1) but hurts under typos on ATIS, SNIPS and MASSIVE (it helps on MIT
  Restaurant). It is opt-in. The external `Entries` list is separate: you supply the
  vocabulary, which is the main way to use this library in a real product.

### Method notes

- Hyperparameters were chosen on dev splits (ATIS dev, MASSIVE dev, CLINC150 val, and 10% holdouts of
  SNIPS and BANKING77) and confirmed with 5-fold cross-validation for the first round. Test sets are used
  for reporting. Differences under about 0.5 points are within noise at these sizes.
- **ATIS taught us a lesson.** Our first TF-IDF default was never worse than the baseline on any dev split,
  yet dropped ATIS test intent accuracy from 95.07% to 91.04%: the ATIS test set is distribution-shifted
  relative to its dev split, and a dominant class (`atis_flight`, 74% of training data) swallowed rare
  intents. We then tried class-balanced loss and weaker row normalisation. They narrow the gap but none
  beats the baseline on ATIS while keeping the many-class gains, so we chose the best overall trade-off.
  **Because we examined ATIS test while diagnosing this, ATIS test is no longer an untouched holdout for
  intent settings**, and the 1.3-point gap should be read with that in mind.
- We fixed a leak in our own gazetteer evaluation: typo-augmented copies of a training utterance
  must share the jackknife fold of their original, or the copy sees its own slot values. Earlier, more
  flattering gazetteer numbers had this leak.
- ATIS multi-intent labels (`a#b`, 15 of 893 test utterances) are kept as separate
  classes; most are unseen in training and count as errors.
- MASSIVE: English (en-US) only, official train/test partitions. CLINC150: in-scope
  intents only (the out-of-scope split is not yet evaluated).

### Regression gate

CI runs `nlubench check`, which scores the default profile on ATIS, SNIPS, CLINC150 and BANKING77 and fails
if any metric drops more than 0.5 points below [`bench/baseline.json`](bench/baseline.json). Training is
deterministic, so the gate does not flake. After an intentional change, refresh the baseline deliberately:

```bash
go run ./cmd/nlubench check -update
```

## Known limitations

- Linear models over word and character n-grams: no semantic understanding, so
  paraphrases with unseen vocabulary hurt.
- The slot tagger is greedy left-to-right, not a CRF.
- English-oriented tokenizer (lower-cases, splits on non-alphanumerics).
- No out-of-scope / unknown-intent rejection yet; `Confidence` is a softmax
  probability and is not calibrated. Do not treat it as a reliable "I don't know" signal.
- Entity linking is edit-distance based: a one-edit typo can land on a different, closely spelled name.

## License

Apache-2.0. See [LICENSE](LICENSE).
