package dataset

import (
	"reflect"
	"strings"
	"testing"

	"github.com/dmavrotas/gonlu/nlu"
)

func TestParseCLINC(t *testing.T) {
	in := `{"train": [["how do i freeze my card", "freeze_account"], ["what is my balance", "balance"]],
	         "test": [["Balance please!", "balance"]], "oos_test": [["tell me a joke", "oos"]]}`
	got, err := ParseCLINC(strings.NewReader(in), "test")
	if err != nil {
		t.Fatal(err)
	}
	want := []nlu.Utterance{{Tokens: []string{"balance", "please"}, Intent: "balance"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v, want %+v", got, want)
	}
	if _, err := ParseCLINC(strings.NewReader(in), "nope"); err == nil {
		t.Error("expected error for unknown split")
	}
}

func TestParseBanking(t *testing.T) {
	in := "text,category\n\"I am still waiting, on my card?\",card_arrival\nWhere is my top-up,top_up_reverted\n"
	got, err := ParseBanking(strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}
	want := []nlu.Utterance{
		{Tokens: []string{"i", "am", "still", "waiting", "on", "my", "card"}, Intent: "card_arrival"},
		{Tokens: []string{"where", "is", "my", "top", "up"}, Intent: "top_up_reverted"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v\nwant %+v", got, want)
	}
}

func TestParseMassive(t *testing.T) {
	in := `{"partition": "train", "intent": "alarm_set", "annot_utt": "wake me up at [time : nine am] on [date : friday]"}
{"partition": "test", "intent": "alarm_query", "annot_utt": "what alarms do i have"}
{"partition": "train", "intent": "weather_query", "annot_utt": "[place_name : new york] weather"}
`
	got, err := ParseMassive(strings.NewReader(in), "train")
	if err != nil {
		t.Fatal(err)
	}
	want := []nlu.Utterance{
		{Tokens: []string{"wake", "me", "up", "at", "nine", "am", "on", "friday"}, Intent: "alarm_set",
			Slots: []nlu.Span{{Start: 4, End: 6, Label: "time"}, {Start: 7, End: 8, Label: "date"}}},
		{Tokens: []string{"new", "york", "weather"}, Intent: "weather_query",
			Slots: []nlu.Span{{Start: 0, End: 2, Label: "place_name"}}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v\nwant %+v", got, want)
	}
}

func TestParseMITBIO(t *testing.T) {
	in := "B-Rating\t2\nI-Rating\tstart\nO\trestaurants\n\nB-Location\tnearby\n"
	got, err := ParseMITBIO(strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}
	want := []nlu.Utterance{
		{Tokens: []string{"2", "start", "restaurants"}, Slots: []nlu.Span{{Start: 0, End: 2, Label: "Rating"}}},
		{Tokens: []string{"nearby"}, Slots: []nlu.Span{{Start: 0, End: 1, Label: "Location"}}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v\nwant %+v", got, want)
	}
	if _, err := ParseMITBIO(strings.NewReader("onlyonefield\n")); err == nil {
		t.Error("expected error for malformed line")
	}
}
