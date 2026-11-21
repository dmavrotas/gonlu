package dataset

import (
	"math/rand"

	"github.com/dmavrotas/gonlu/nlu"
)

// Perturb returns a copy of data in which each token longer than three bytes is,
// with probability rate, corrupted by a deletion or an adjacent transposition.
// Intents, slot spans and token counts are preserved, so gold labels stay valid.
// The same seed always yields the same output. It is used to measure robustness
// to typos.
func Perturb(data []nlu.Utterance, rate float64, seed int64) []nlu.Utterance {
	r := rand.New(rand.NewSource(seed)) //nolint:gosec // reproducible noise, not security
	out := make([]nlu.Utterance, len(data))
	for i, u := range data {
		toks := append([]string(nil), u.Tokens...)
		for j, t := range toks {
			if len(t) <= 3 || r.Float64() >= rate {
				continue
			}
			p := 1 + r.Intn(len(t)-2) // interior position
			if r.Intn(2) == 0 {
				toks[j] = t[:p] + t[p+1:]
			} else {
				b := []byte(t)
				b[p], b[p-1] = b[p-1], b[p]
				toks[j] = string(b)
			}
		}
		out[i] = nlu.Utterance{Tokens: toks, Intent: u.Intent, Slots: u.Slots}
	}
	return out
}
