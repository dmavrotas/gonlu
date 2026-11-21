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

// SpansFromBIO converts BIO tags ("B-x", "I-x", "O") to spans. An I- tag whose
// label differs from the open span, or that has no open span, starts a new one.
func SpansFromBIO(tags []string) []Span {
	var spans []Span
	start, label := -1, ""
	flush := func(end int) {
		if start >= 0 {
			spans = append(spans, Span{Start: start, End: end, Label: label})
			start = -1
		}
	}
	for i, tag := range tags {
		switch {
		case strings.HasPrefix(tag, "B-"):
			flush(i)
			start, label = i, tag[2:]
		case strings.HasPrefix(tag, "I-") && start >= 0 && tag[2:] == label:
		case strings.HasPrefix(tag, "I-"):
			flush(i)
			start, label = i, tag[2:]
		default:
			flush(i)
		}
	}
	flush(len(tags))
	return spans
}

// BIOFromSpans is the inverse of SpansFromBIO for n tokens. Overlapping or
// out-of-range spans are ignored.
func BIOFromSpans(n int, spans []Span) []string {
	tags := make([]string, n)
	for i := range tags {
		tags[i] = "O"
	}
	for _, s := range spans {
		if s.Start < 0 || s.End > n || s.Start >= s.End {
			continue
		}
		free := true
		for i := s.Start; i < s.End; i++ {
			if tags[i] != "O" {
				free = false
			}
		}
		if !free {
			continue
		}
		tags[s.Start] = "B-" + s.Label
		for i := s.Start + 1; i < s.End; i++ {
			tags[i] = "I-" + s.Label
		}
	}
	return tags
}
