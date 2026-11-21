package dataset

import (
	"reflect"
	"strings"
	"testing"

	"github.com/dmavrotas/nlu/nlu"
)

func TestParseATIS(t *testing.T) {
	in := "BOS flights from boston to denver EOS\t O O O B-fromloc.city_name O B-toloc.city_name atis_flight\n" +
		"BOS san francisco EOS\t O B-fromloc.city_name I-fromloc.city_name atis_airfare#atis_flight\n"
	got, err := ParseATIS(strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}
	want := []nlu.Utterance{
		{Tokens: []string{"flights", "from", "boston", "to", "denver"}, Intent: "atis_flight",
			Slots: []nlu.Span{{Start: 2, End: 3, Label: "fromloc.city_name"}, {Start: 4, End: 5, Label: "toloc.city_name"}}},
		{Tokens: []string{"san", "francisco"}, Intent: "atis_airfare#atis_flight",
			Slots: []nlu.Span{{Start: 0, End: 2, Label: "fromloc.city_name"}}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v\nwant %+v", got, want)
	}
}

func TestParseATISMalformed(t *testing.T) {
	if _, err := ParseATIS(strings.NewReader("no tab here\n")); err == nil {
		t.Error("expected error for line without tab")
	}
	if _, err := ParseATIS(strings.NewReader("BOS a b EOS\t O O atis_x\n")); err == nil {
		t.Error("expected error for tag/token count mismatch")
	}
}

func TestParseSnips(t *testing.T) {
	in := `{"PlayMusic":[{"data":[{"text":"play "},{"text":"Sam Smith","entity":"artist"},{"text":" on spotify"}]}]}`
	got, err := ParseSnips(strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}
	want := []nlu.Utterance{{
		Tokens: []string{"play", "sam", "smith", "on", "spotify"},
		Intent: "PlayMusic",
		Slots:  []nlu.Span{{Start: 1, End: 3, Label: "artist"}},
	}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v\nwant %+v", got, want)
	}
}
