package nlu

import (
	"reflect"
	"testing"
)

func TestParseAnnotated(t *testing.T) {
	tests := []struct {
		name, in string
		want     Utterance
	}{
		{"two slots", "wake me up at [time : nine am] on [date : friday]", Utterance{
			Tokens: []string{"wake", "me", "up", "at", "nine", "am", "on", "friday"}, Intent: "alarm",
			Slots: []Span{{Start: 4, End: 6, Label: "time"}, {Start: 7, End: 8, Label: "date"}}}},
		{"slot first", "[city : new york] weather", Utterance{
			Tokens: []string{"new", "york", "weather"}, Intent: "alarm",
			Slots: []Span{{Start: 0, End: 2, Label: "city"}}}},
		{"no slots", "what time is it", Utterance{Tokens: []string{"what", "time", "is", "it"}, Intent: "alarm"}},
		{"tight brackets", "play [artist:queen]", Utterance{
			Tokens: []string{"play", "queen"}, Intent: "alarm",
			Slots: []Span{{Start: 1, End: 2, Label: "artist"}}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ParseAnnotated(tt.in, "alarm"); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("got %+v\nwant %+v", got, tt.want)
			}
		})
	}
}
