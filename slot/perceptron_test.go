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
