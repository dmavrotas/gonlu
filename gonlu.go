// Package gonlu is a small, deterministic, offline natural-language-understanding
// toolkit: it classifies the intent of a sentence and extracts its slots
// (entities), with no network calls, no cgo and no model files.
//
// Train a model from annotated examples and parse new text:
//
//	data := []nlu.Utterance{
//		nlu.ParseAnnotated("book a table in [city : rome] for [party : two]", "book_table"),
//		nlu.ParseAnnotated("what is the weather in [city : paris]", "get_weather"),
//		// ... more examples
//	}
//	model, err := gonlu.Train(data, gonlu.Options{})
//	res := model.Parse("book a table in oslo for four")
//	// res.Intent == "book_table"; res.Slots holds city=oslo, party=four
package gonlu

import (
	"errors"
	"strings"

	"github.com/dmavrotas/gonlu/intent"
	"github.com/dmavrotas/gonlu/nlu"
	"github.com/dmavrotas/gonlu/slot"
)

// Options configures Train. The zero value gives the benchmark-default model.
type Options struct {
	// Robust trains for noisy, human-typed input by adding typo-augmented
	// training data. It costs little on clean text and recovers a lot on text
	// with misspellings.
	Robust bool
	// Entries supplies known entity values per slot type, for example
	// {"driver": {"Lewis Hamilton", "Max Verstappen"}}. They are matched
	// typo-tolerantly, help label values never seen in the training data, and
	// are used to link each extracted slot to its canonical entry (Slot.Entry).
	// List more important entries first: they win ties.
	Entries map[string][]string
	// Epochs overrides the number of slot-tagger training epochs (default 10).
	Epochs int
}

// Slot is one extracted entity.
type Slot struct {
	Label      string // slot type, for example "city"
	Text       string // the matched tokens joined by spaces (lower-cased)
	Start, End int    // token offsets into the tokenised input; End is exclusive
	// Entry is the closest value from Options.Entries for this slot's type,
	// in its original spelling (for example "Max Verstappen" for the text
	// "verstapen"). It is empty when no entry is close enough or the type has no
	// entries. When several entries match equally well, the one listed first wins.
	Entry string
}

// Result is the parse of one sentence.
type Result struct {
	Intent     string  // empty when the model was trained without intents
	Confidence float64 // probability of Intent in (0,1]; 0 when there is no intent
	Tokens     []string
	Slots      []Slot
}

// Model is a trained intent classifier and slot tagger. It is safe for
// concurrent use after Train returns.
type Model struct {
	clf    *intent.LogReg
	tagger *slot.Tagger
	link   *linker
}

// ErrNoData is returned by Train when there are no training utterances.
var ErrNoData = errors.New("gonlu: no training data")

// Train fits a model. Utterances with an empty Intent are ignored by the intent
// classifier, and utterances without slots simply teach the tagger that
// nothing should be extracted.
func Train(data []nlu.Utterance, opts Options) (*Model, error) {
	if len(data) == 0 {
		return nil, ErrNoData
	}
	lo := intent.DefaultOptions()
	so := slot.Options{Epochs: opts.Epochs, Entries: opts.Entries}
	if opts.Robust {
		lo = intent.RobustOptions()
		so.CharFeatures, so.TypoAugment = true, 0.3
	}

	var ex []intent.Example
	for _, u := range data {
		if u.Intent != "" {
			ex = append(ex, intent.Example{Tokens: u.Tokens, Label: u.Intent})
		}
	}
	return &Model{clf: intent.TrainLogReg(ex, lo), tagger: slot.TrainOptions(data, so), link: newLinker(opts.Entries)}, nil
}

// Parse classifies text and extracts its slots.
func (m *Model) Parse(text string) Result {
	toks := nlu.Tokenize(text)
	res := Result{Tokens: toks}
	if len(toks) == 0 {
		return res
	}
	if m.clf != nil {
		res.Intent, res.Confidence = m.clf.Predict(toks)
	}
	for _, s := range m.tagger.Tag(toks) {
		text := strings.Join(toks[s.Start:s.End], " ")
		res.Slots = append(res.Slots, Slot{Label: s.Label, Text: text, Start: s.Start, End: s.End, Entry: m.link.link(s.Label, text)})
	}
	return res
}

// Link resolves text to the closest value in Options.Entries for the given slot
// type, the same lookup Parse uses to fill Slot.Entry. It also returns the match
// cost, lower being better: 0 for the whole entry, 1 for its last word (a
// surname), 2 for any earlier word (a forename), and each typo edit adds 3.
// ok is false when the type has no entries or nothing is close enough.
//
// Parse links a slot only within the type the tagger chose. Link lets a caller
// compare several types and let the name itself decide, which is more reliable
// than the tagger's guess when a sentence frame fits more than one type.
func (m *Model) Link(slotType, text string) (entry string, cost int, ok bool) {
	return m.link.best(slotType, strings.TrimSpace(strings.ToLower(text)))
}
