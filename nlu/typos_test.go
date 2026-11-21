package nlu

import (
	"reflect"
	"testing"
)

func sample() []Utterance {
	return []Utterance{
		{Tokens: []string{"show", "flights", "from", "boston", "to", "denver"}, Intent: "flight",
			Slots: []Span{{Start: 3, End: 4, Label: "from"}}},
	}
}

func TestAddTyposZeroRateIsIdentity(t *testing.T) {
	in := sample()
	if got := AddTypos(in, 0, 1); !reflect.DeepEqual(got, in) {
		t.Errorf("rate 0 changed data: %v", got)
	}
}

func TestAddTyposFullRateChangesLongWordsOnly(t *testing.T) {
	in := sample()
	got := AddTypos(in, 1, 1)[0]
	for i, tok := range in[0].Tokens {
		if len(tok) <= 3 && got.Tokens[i] != tok {
			t.Errorf("short token %q was changed to %q", tok, got.Tokens[i])
		}
	}
	changed := 0
	for i, tok := range in[0].Tokens {
		if len(tok) > 3 && got.Tokens[i] != tok {
			changed++
		}
	}
	if changed == 0 {
		t.Error("no long token was perturbed at rate 1")
	}
	if got.Intent != in[0].Intent || !reflect.DeepEqual(got.Slots, in[0].Slots) || len(got.Tokens) != len(in[0].Tokens) {
		t.Error("intent, slots and token count must be preserved")
	}
}

func TestAddTyposDeterministicAndNonMutating(t *testing.T) {
	in := sample()
	orig := append([]string(nil), in[0].Tokens...)
	a, b := AddTypos(in, 0.5, 42), AddTypos(in, 0.5, 42)
	if !reflect.DeepEqual(a, b) {
		t.Error("same seed must give same output")
	}
	if !reflect.DeepEqual(in[0].Tokens, orig) {
		t.Error("input was mutated")
	}
}
