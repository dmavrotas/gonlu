package slot

import (
	"reflect"
	"testing"

	"github.com/dmavrotas/gonlu/nlu"
)

func utt(text string, slots ...nlu.Span) nlu.Utterance {
	return nlu.Utterance{Tokens: nlu.Tokenize(text), Slots: slots}
}

func span(s, e int, l string) nlu.Span { return nlu.Span{Start: s, End: e, Label: l} }

func trainingSet() []nlu.Utterance {
	return []nlu.Utterance{
		utt("flights from boston to denver", span(2, 3, "from"), span(4, 5, "to")),
		utt("flights from denver to boston", span(2, 3, "from"), span(4, 5, "to")),
		utt("show flights from dallas to atlanta", span(3, 4, "from"), span(5, 6, "to")),
		utt("flights from san francisco to boston", span(2, 4, "from"), span(5, 6, "to")),
		utt("flights from atlanta to new york", span(2, 3, "from"), span(4, 6, "to")),
		utt("list flights to denver", span(3, 4, "to")),
		utt("flights from pittsburgh", span(2, 3, "from")),
	}
}

func TestTaggerLearnsTrainingData(t *testing.T) {
	data := trainingSet()
	tg := Train(data, 10)
	for _, u := range data {
		if got := tg.Tag(u.Tokens); !reflect.DeepEqual(got, u.Slots) {
			t.Errorf("Tag(%v) = %v, want %v", u.Tokens, got, u.Slots)
		}
	}
}

func TestTaggerGeneralisesFromContext(t *testing.T) {
	tg := Train(trainingSet(), 10)
	// "tulsa" and "omaha" were never seen: only context can label them.
	got := tg.Tag(nlu.Tokenize("flights from tulsa to omaha"))
	want := []nlu.Span{span(2, 3, "from"), span(4, 5, "to")}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestTaggerDeterministic(t *testing.T) {
	a := Train(trainingSet(), 5).Tag(nlu.Tokenize("flights from tulsa to omaha"))
	b := Train(trainingSet(), 5).Tag(nlu.Tokenize("flights from tulsa to omaha"))
	if !reflect.DeepEqual(a, b) {
		t.Errorf("training is not deterministic: %v vs %v", a, b)
	}
}

func TestTaggerEdgeCases(t *testing.T) {
	tg := Train(trainingSet(), 3)
	if got := tg.Tag(nil); got != nil {
		t.Errorf("Tag(nil) = %v, want nil", got)
	}
	if Train(nil, 3) != nil {
		t.Error("Train(nil) should return nil")
	}
}

func TestTaggerCharFeaturesHandleUnseenWordForms(t *testing.T) {
	tg := TrainOptions(trainingSet(), Options{Epochs: 10, CharFeatures: true})
	got := tg.Tag(nlu.Tokenize("flights from tulsa to omaha"))
	want := []nlu.Span{span(2, 3, "from"), span(4, 5, "to")}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestTaggerTypoAugmentSurvivesMisspelledContext(t *testing.T) {
	// "flihgts" / "frmo" never appear in training; augmentation plus char
	// features should keep the slots intact.
	tg := TrainOptions(trainingSet(), Options{Epochs: 10, CharFeatures: true, TypoAugment: 0.5})
	got := tg.Tag(nlu.Tokenize("flights from tulsa to omaha"))
	want := []nlu.Span{span(2, 3, "from"), span(4, 5, "to")}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("clean input regressed with augmentation: got %v, want %v", got, want)
	}
}

func TestTrainOptionsDoesNotMutateInput(t *testing.T) {
	data := trainingSet()
	before := len(data)
	first := append([]string(nil), data[0].Tokens...)
	TrainOptions(data, Options{TypoAugment: 1})
	if len(data) != before || !reflect.DeepEqual(data[0].Tokens, first) {
		t.Error("training data was modified")
	}
}

func TestGazetteerEntriesLabelUnseenNames(t *testing.T) {
	data := []nlu.Utterance{
		utt("book a flight for alice", span(4, 5, "name")),
		utt("book a flight for bob", span(4, 5, "name")),
		utt("book a flight for carol", span(4, 5, "name")),
		utt("book a flight for dave", span(4, 5, "name")),
		utt("book a flight for erin", span(4, 5, "name")),
		utt("book a flight for frank", span(4, 5, "name")),
		utt("flight to paris", span(2, 3, "city")),
		utt("flight to rome", span(2, 3, "city")),
		utt("flight to berlin", span(2, 3, "city")),
		utt("flight to madrid", span(2, 3, "city")),
	}
	entries := map[string][]string{"city": {"Lisbon", "Vienna"}}
	tg := TrainOptions(data, Options{Epochs: 15, Entries: entries})
	// "lisbon" never occurs in training; only the external entry can mark it a city.
	got := tg.Tag(nlu.Tokenize("flight to lisbon"))
	want := []nlu.Span{span(2, 3, "city")}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
	// One-letter typo is still recognised through the fuzzy index.
	got = tg.Tag(nlu.Tokenize("flight to vienan"))
	want = []nlu.Span{span(2, 3, "city")}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("typo: got %v, want %v", got, want)
	}
}

func TestLexiconFuzzyMatching(t *testing.T) {
	lx := buildLexicon(nil, map[string][]string{"driver": {"Hamilton"}})
	tests := []struct {
		word string
		want []string
	}{
		{"hamilton", []string{"gz:driver"}},
		{"hamiltno", []string{"gzf:driver"}}, // transposition
		{"hamilto", []string{"gzf:driver"}},  // deletion
		{"hamiltxn", []string{"gzf:driver"}}, // substitution
		{"ham", nil},                         // too short to fuzzy match
		{"verstappen", nil},
	}
	for _, tt := range tests {
		if got := lx.features(tt.word); !reflect.DeepEqual(got, tt.want) {
			t.Errorf("features(%q) = %v, want %v", tt.word, got, tt.want)
		}
	}
}

func TestJackknifeCopiesShareOriginalFold(t *testing.T) {
	orig := []nlu.Utterance{utt("a x", span(1, 2, "t")), utt("b y", span(1, 2, "t")), utt("c z", span(1, 2, "t"))}
	data := append(append([]nlu.Utterance(nil), orig...), orig...) // exact copies
	gaz := jackknife(data, len(orig), nil)
	for i := range orig {
		if !reflect.DeepEqual(gaz[i], gaz[i+len(orig)]) {
			t.Errorf("copy %d saw a different lexicon than its original", i)
		}
		// the value's own word must not appear as a known word for its own utterance
		for _, f := range gaz[i][1] {
			if f == "gz:t" {
				t.Errorf("utterance %d saw its own slot value in the lexicon", i)
			}
		}
	}
}
