package nlu

import (
	"reflect"
	"testing"
)

func TestTokenize(t *testing.T) {
	tests := []struct {
		name, in string
		want     []string
	}{
		{"lowercases and splits", "Play Some Jazz!", []string{"play", "some", "jazz"}},
		{"keeps apostrophes", "What's the weather", []string{"what's", "the", "weather"}},
		{"keeps digits", "at 8:38 am", []string{"at", "8", "38", "am"}},
		{"empty", "  ", []string{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Tokenize(tt.in); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Tokenize(%q) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

func TestBIORoundTrip(t *testing.T) {
	spans := []Span{{Start: 1, End: 3, Label: "city"}, {Start: 4, End: 5, Label: "date"}}
	tags := BIOFromSpans(6, spans)
	want := []string{"O", "B-city", "I-city", "O", "B-date", "O"}
	if !reflect.DeepEqual(tags, want) {
		t.Fatalf("BIOFromSpans = %v, want %v", tags, want)
	}
	if got := SpansFromBIO(tags); !reflect.DeepEqual(got, spans) {
		t.Errorf("SpansFromBIO = %v, want %v", got, spans)
	}
}

func TestSpansFromBIOEdgeCases(t *testing.T) {
	tests := []struct {
		name string
		in   []string
		want []Span
	}{
		{"stray I starts span", []string{"I-a", "I-a"}, []Span{{Start: 0, End: 2, Label: "a"}}},
		{"label change splits", []string{"B-a", "I-b"}, []Span{{Start: 0, End: 1, Label: "a"}, {Start: 1, End: 2, Label: "b"}}},
		{"adjacent B splits", []string{"B-a", "B-a"}, []Span{{Start: 0, End: 1, Label: "a"}, {Start: 1, End: 2, Label: "a"}}},
		{"all O", []string{"O", "O"}, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := SpansFromBIO(tt.in); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("got %v want %v", got, tt.want)
			}
		})
	}
}

func TestBIOFromSpansIgnoresBadSpans(t *testing.T) {
	got := BIOFromSpans(3, []Span{{Start: 0, End: 2, Label: "a"}, {Start: 1, End: 3, Label: "b"}, {Start: 2, End: 9, Label: "c"}})
	want := []string{"B-a", "I-a", "O"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v want %v", got, want)
	}
}
