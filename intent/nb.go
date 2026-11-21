// Package intent provides intent classifiers.
package intent

import (
	"math"
	"sort"
)

// Example is one labelled training utterance.
type Example struct {
	Tokens []string
	Label  string
}

// NaiveBayes is a multinomial Naive Bayes classifier over word unigrams and
// bigrams with Laplace smoothing. It is the baseline every later model must beat.
type NaiveBayes struct {
	labels   []string
	logPrior map[string]float64
	counts   map[string]map[string]float64 // label -> feature -> count
	totals   map[string]float64            // label -> total feature count
	vocab    int
}

// features returns unigrams plus bigrams, with sentence boundary markers.
func features(tokens []string) []string {
	f := make([]string, 0, 2*len(tokens)+1)
	prev := "<s>"
	for _, t := range tokens {
		f = append(f, t, prev+" "+t)
		prev = t
	}
	return append(f, prev+" </s>")
}

// TrainNaiveBayes fits a classifier. It returns nil for empty input.
func TrainNaiveBayes(examples []Example) *NaiveBayes {
	if len(examples) == 0 {
		return nil
	}
	nb := &NaiveBayes{
		logPrior: map[string]float64{},
		counts:   map[string]map[string]float64{},
		totals:   map[string]float64{},
	}
	docs := map[string]float64{}
	vocab := map[string]struct{}{}
	for _, ex := range examples {
		docs[ex.Label]++
		if nb.counts[ex.Label] == nil {
			nb.counts[ex.Label] = map[string]float64{}
		}
		for _, f := range features(ex.Tokens) {
			nb.counts[ex.Label][f]++
			nb.totals[ex.Label]++
			vocab[f] = struct{}{}
		}
	}
	nb.vocab = len(vocab)
	for l, n := range docs {
		nb.labels = append(nb.labels, l)
		nb.logPrior[l] = math.Log(n / float64(len(examples)))
	}
	sort.Strings(nb.labels) // deterministic tie-breaking
	return nb
}

// Predict returns the most likely label and its posterior probability.
func (nb *NaiveBayes) Predict(tokens []string) (label string, confidence float64) {
	feats := features(tokens)
	scores := make([]float64, len(nb.labels))
	best := 0
	for i, l := range nb.labels {
		s := nb.logPrior[l]
		denom := nb.totals[l] + float64(nb.vocab)
		for _, f := range feats {
			s += math.Log((nb.counts[l][f] + 1) / denom)
		}
		scores[i] = s
		if s > scores[best] {
			best = i
		}
	}
	// softmax over log scores for a calibrated-ish confidence
	var sum float64
	for _, s := range scores {
		sum += math.Exp(s - scores[best])
	}
	return nb.labels[best], 1 / sum
}
