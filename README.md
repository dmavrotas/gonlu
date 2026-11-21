# gonlu

A small, deterministic, **offline, pure-Go** natural-language-understanding toolkit.
It classifies the **intent** of a sentence and extracts its **slots** (entities),
with no network calls, no cgo, no model files and no external services.

It was extracted from [pitwall](https://github.com/dmavrotas/pitwall), a rule-based
natural-language query engine for Formula 1 data, and is developed against public
benchmarks (ATIS, SNIPS, MASSIVE, CLINC150, BANKING77, MIT Restaurant).

> Status: early (v0). The API may change. Accuracy is that of a linear model: it is
> strong on small, domain-specific vocabularies and clearly behind fine-tuned
> transformers on large, fine-grained intent sets (see [Benchmarks](#benchmarks)).

## Install

```bash
go get github.com/dmavrotas/gonlu
```

## Example 1: train on a handful of sentences

Write training data with inline `[label : value]` markup, train, parse.
([`examples/quickstart`](examples/quickstart/main.go))

```go
var data []nlu.Utterance
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
  intent: get_weather (100%)
  slot:   city = "oslo"
"reserve a table for four people in Lisbon"
  intent: book_table (100%)
  slot:   party = "four"
```

Note the last line: `lisbon` is missed. With only 24 training sentences the model never saw
a city after "people in". More examples fix this; so does supplying known values
(next example).

## Example 2: plug in your own vocabulary, survive typos

`Entries` gives the model known entity values it never saw in training, matched
typo-tolerantly. `Robust` trains for human-typed input.
([`examples/entities`](examples/entities/main.go))

```go
entries := map[string][]string{
	"driver": {"Lewis Hamilton", "Max Verstappen", "Charles Leclerc", "Fernando Alonso"},
}
model, _ := gonlu.Train(data, gonlu.Options{Robust: true, Entries: entries})
```

Real output of `go run ./examples/entities`. Verstappen, Hamilton and Leclerc appear
nowhere in the training data (which only mentions senna, prost, alonso and a few
others); they come from `Entries`:

```
"how many wins does verstappen have"     -> wins    driver="verstappen"
"how many poles did hamilton take"       -> poles   driver="hamilton"
"how many poles did hamliton take"       -> poles   driver="hamliton"
"how many wnis does leclerc have"        -> wins    driver="leclerc"
```

The last two contain typos (`hamliton`, `wnis`) and are still understood.

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
  intent: GetWeather (100%)
  slot:   city                     "paris tomorrow"     <- "tomorrow" should be separate
"Book a table for 4 at a pizza place in Boston"
  intent: BookRestaurant (100%)
  slot:   party_size_number        "4"
  slot:   served_dish              "pizza"
  slot:   city                     "boston"
"Add Bohemian Rhapsody to my road trip playlist"
  intent: AddToPlaylist (100%)
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
  intent: SearchScreeningEvent (100%)
  slot:   movie_name               "black panther"
  slot:   object_location_type     "cinema"             <- "rialto" (the venue) was missed
```

(The `<-` notes are ours, not program output.) All six intents are right; slot
filling is good but not perfect, which matches the benchmark numbers below.

## Packages

| Package | What it is |
|---|---|
| `gonlu` | `Train` / `Parse`: the simple entry point |
| `nlu` | core types (`Utterance`, `Span`), `Tokenize`, `ParseAnnotated`, BIO helpers, `AddTypos` |
| `intent` | intent classifiers: multinomial logistic regression (Adagrad), Naive Bayes baseline |
| `slot` | slot tagger: greedy averaged-perceptron BIO tagger with optional char features, typo augmentation and a typo-tolerant gazetteer |
| `bench/...` | dataset loaders and metrics (intent accuracy, exact-match slot F1, frame accuracy) |
| `cmd/nlubench` | downloads the public benchmarks and scores the library |

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

Three profiles are reported. `default` is word uni/bigrams. `robust` adds character
trigrams and typo-augmented training data (`Options{Robust: true}`). `robust+gaz`
adds the learned gazetteer on top (`slot.Options{Gazetteer: true}`).

### Clean test sets

| Dataset | Task | default | robust | robust+gaz |
|---|---|---|---|---|
| ATIS | intent / slot F1 / frame | **95.07** / 93.27 / **80.29** | 93.62 / 93.35 / 79.73 | 93.62 / **93.75** / 80.18 |
| SNIPS | intent / slot F1 / frame | 98.00 / 91.49 / 79.71 | 98.00 / 91.17 / 79.14 | 98.00 / **92.32** / **80.43** |
| MASSIVE (en-US) | intent / slot F1 / frame | **81.88** / **76.40** / **61.90** | 79.52 / 75.99 / 60.63 | 79.52 / 76.29 / 60.79 |
| CLINC150 | intent | 86.49 | **88.11** | - |
| BANKING77 | intent | 85.75 | **86.40** | - |
| MIT Restaurant | slot F1 | 76.28 | 76.13 | **77.23** |

### With 30% of test words corrupted (deletion or transposition)

| Dataset | Task | default | robust | robust+gaz |
|---|---|---|---|---|
| ATIS | intent / slot F1 / frame | 86.90 / 87.32 / 57.89 | **92.39** / **92.42** / **75.81** | 92.39 / 91.74 / 73.01 |
| SNIPS | intent / slot F1 / frame | 95.00 / 80.18 / 58.00 | **97.14** / **89.40** / **75.00** | 97.14 / 87.02 / 69.29 |
| MASSIVE (en-US) | intent / slot F1 / frame | 69.77 / 63.96 / 45.66 | **74.68** / **72.28** / **53.97** | 74.68 / 71.85 / 53.97 |
| CLINC150 | intent | 70.53 | **83.53** | - |
| BANKING77 | intent | 71.69 | **80.94** | - |
| MIT Restaurant | slot F1 | 63.13 | 73.58 | **74.52** |

All numbers are percentages on the official test splits (SNIPS: the validation set
of the 2017 benchmark). Slot F1 is exact span-and-label match, micro-averaged.

### How to read these

- **Where it is strong.** Small, domain-specific vocabularies: ATIS and SNIPS intents
  are 95-98%. Neural joint models in the literature report about 98% intent accuracy and
  96% slot F1 on ATIS, so on ATIS we are within about 3 points for intent and
  2-3 for slots, with no model files.
- **Where it is weak.** Large, fine-grained intent sets: CLINC150 (150 classes) and
  BANKING77 (77 classes) sit in the mid-to-high 80s, where fine-tuned transformers
  reach the 90s. We have not yet compared against published numbers for MASSIVE,
  CLINC150, BANKING77 or MIT Restaurant, so treat those as our own measurements, not a
  leaderboard claim.
- **Robust mode** costs 0.0-2.4 points on clean text but recovers 2-13 points when
  words are misspelled. It even helps clean CLINC150 and BANKING77. Use it for
  human-typed input.
- **The learned gazetteer** is a small, consistent gain on clean slot filling (+0.3 to
  +1.2 F1) but hurts under typos on ATIS, SNIPS and MASSIVE (it helps on MIT
  Restaurant). It is opt-in. The external `Entries` list is separate: you supply the
  vocabulary, which is the main way to use this library in a real product.

### Method notes

- Hyperparameters were chosen on the ATIS dev split and confirmed with 5-fold
  cross-validation. The test sets are used only for reporting. Differences under about
  0.5 points are within noise at these sizes.
- We fixed a leak in our own gazetteer evaluation: typo-augmented copies of a
  training utterance must share the jackknife fold of their original, or the copy sees
  its own slot values. Earlier, more flattering gazetteer numbers had this leak.
- ATIS multi-intent labels (`a#b`, 15 of 893 test utterances) are kept as separate
  classes; most are unseen in training and count as errors.
- MASSIVE: English (en-US) only, official train/test partitions. CLINC150: in-scope
  intents only (the out-of-scope split is not yet evaluated).

## Known limitations

- Linear models over word and character n-grams: no semantic understanding, so
  paraphrases with unseen vocabulary hurt (see CLINC150 and BANKING77).
- The slot tagger is greedy left-to-right, not a CRF.
- English-oriented tokenizer (lower-cases, splits on non-alphanumerics).
- No out-of-scope / unknown-intent rejection yet; `Confidence` is a softmax
  probability and is not calibrated.

## License

Apache-2.0. See [LICENSE](LICENSE).
