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
