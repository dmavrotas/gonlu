// Package nlu defines the core data types shared by the library's classifiers,
// taggers and benchmark harness.
package nlu

import (
	"strings"
	"unicode"
)

// Span marks a labelled run of tokens. Start is inclusive, End exclusive.
type Span struct {
	Start, End int
	Label      string
}

// Utterance is a tokenised sentence with an intent label and slot spans.
type Utterance struct {
	Tokens []string
	Intent string
	Slots  []Span
}

// Tokenize lowercases text and splits it into word tokens. Letters, digits and
// apostrophes are kept inside tokens; everything else separates them.
func Tokenize(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '\''
	})
}
