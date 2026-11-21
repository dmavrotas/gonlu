// Package metrics implements the standard NLU evaluation measures: intent
// accuracy, exact-match slot-span F1 (conlleval-style) and frame accuracy.
package metrics

import (
	"sort"

	"github.com/dmavrotas/gonlu/nlu"
)

// PRF holds precision, recall and F1, each in [0,1].
type PRF struct{ Precision, Recall, F1 float64 }

// Accuracy returns the fraction of positions where got equals want.
func Accuracy(got, want []string) float64 {
	if len(want) == 0 {
		return 0
	}
	hit := 0
	for i := range want {
		if i < len(got) && got[i] == want[i] {
			hit++
		}
	}
	return float64(hit) / float64(len(want))
}

func key(s nlu.Span) nlu.Span { return s }

// SlotF1 computes micro-averaged span F1. A predicted span counts only if
// start, end and label all match a gold span exactly.
func SlotF1(got, want [][]nlu.Span) PRF {
	var tp, nGot, nWant int
	for i := range want {
		gold := make(map[nlu.Span]bool, len(want[i]))
		for _, s := range want[i] {
			gold[key(s)] = true
		}
		nWant += len(want[i])
		if i >= len(got) {
			continue
		}
		nGot += len(got[i])
		for _, s := range got[i] {
			if gold[key(s)] {
				tp++
			}
		}
	}
	var r PRF
	if nGot > 0 {
		r.Precision = float64(tp) / float64(nGot)
	}
	if nWant > 0 {
		r.Recall = float64(tp) / float64(nWant)
	}
	if r.Precision+r.Recall > 0 {
		r.F1 = 2 * r.Precision * r.Recall / (r.Precision + r.Recall)
	}
	return r
}

func sameSpans(a, b []nlu.Span) bool {
	if len(a) != len(b) {
		return false
	}
	a, b = append([]nlu.Span(nil), a...), append([]nlu.Span(nil), b...)
	less := func(s []nlu.Span) func(i, j int) bool {
		return func(i, j int) bool {
			if s[i].Start != s[j].Start {
				return s[i].Start < s[j].Start
			}
			return s[i].End < s[j].End
		}
	}
	sort.Slice(a, less(a))
	sort.Slice(b, less(b))
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// FrameAccuracy is the fraction of utterances where the intent and the full
// set of slots are both exactly right.
func FrameAccuracy(gotIntent, wantIntent []string, got, want [][]nlu.Span) float64 {
	if len(wantIntent) == 0 {
		return 0
	}
	hit := 0
	for i := range wantIntent {
		if i < len(gotIntent) && i < len(got) && gotIntent[i] == wantIntent[i] && sameSpans(got[i], want[i]) {
			hit++
		}
	}
	return float64(hit) / float64(len(wantIntent))
}
