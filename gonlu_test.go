package gonlu

import (
	"reflect"
	"testing"

	"github.com/dmavrotas/gonlu/nlu"
)

func trainingData() []nlu.Utterance {
	d := make([]nlu.Utterance, 0, 24)
	for _, c := range []string{"rome", "paris", "berlin", "madrid", "oslo", "lima"} {
		d = append(d,
			nlu.ParseAnnotated("book a table in [city : "+c+"] for [party : two]", "book_table"),
			nlu.ParseAnnotated("reserve a table in [city : "+c+"] for [party : four] people", "book_table"),
			nlu.ParseAnnotated("what is the weather in [city : "+c+"]", "get_weather"),
			nlu.ParseAnnotated("will it rain in [city : "+c+"] tomorrow", "get_weather"),
		)
	}
	return d
}

func TestTrainAndParse(t *testing.T) {
	m, err := Train(trainingData(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	got := m.Parse("Book a table in Vienna for two")
	if got.Intent != "book_table" {
		t.Errorf("intent = %q, want book_table", got.Intent)
	}
	if got.Confidence <= 0.5 || got.Confidence > 1 {
		t.Errorf("confidence %v out of range", got.Confidence)
	}
	want := []Slot{{Label: "city", Text: "vienna", Start: 4, End: 5}, {Label: "party", Text: "two", Start: 6, End: 7}}
	if !reflect.DeepEqual(got.Slots, want) {
		t.Errorf("slots = %+v, want %+v", got.Slots, want)
	}
}

func TestTrainNoData(t *testing.T) {
	if _, err := Train(nil, Options{}); err == nil {
		t.Error("expected error for empty training data")
	}
}

func TestSlotOnlyData(t *testing.T) {
	d := []nlu.Utterance{
		nlu.ParseAnnotated("[x : a1] foo", ""), nlu.ParseAnnotated("[x : b2] foo", ""),
		nlu.ParseAnnotated("[x : c3] foo", ""), nlu.ParseAnnotated("bar baz", ""),
	}
	m, err := Train(d, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if r := m.Parse("zz foo"); r.Intent != "" || r.Confidence != 0 {
		t.Errorf("slot-only model returned intent %q (%v)", r.Intent, r.Confidence)
	}
}

func TestEntriesAndRobustMode(t *testing.T) {
	m, err := Train(trainingData(), Options{Robust: true, Entries: map[string][]string{"city": {"Lisbon", "Vienna"}}})
	if err != nil {
		t.Fatal(err)
	}
	got := m.Parse("wether in lisbon")
	if got.Intent != "get_weather" {
		t.Errorf("typo intent = %q, want get_weather", got.Intent)
	}
	if len(got.Slots) != 1 || got.Slots[0].Label != "city" || got.Slots[0].Text != "lisbon" {
		t.Errorf("slots = %+v", got.Slots)
	}
}

func TestParseEmpty(t *testing.T) {
	m, _ := Train(trainingData(), Options{})
	if r := m.Parse("  !! "); len(r.Slots) != 0 {
		t.Errorf("empty input produced slots: %+v", r.Slots)
	}
}
