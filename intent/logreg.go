package intent

import (
	"math"
	"math/rand"
	"sort"

	"github.com/dmavrotas/gonlu/nlu"
)

// Classifier predicts an intent label and a confidence in (0,1] for tokens.
type Classifier interface {
	Predict(tokens []string) (label string, confidence float64)
}

// LogRegOptions tunes logistic-regression training. Zero values pick defaults.
type LogRegOptions struct {
	Epochs       int     // default 20
	LearningRate float64 // Adagrad base rate; default 0.5
	L2           float64 // weight decay on active features; default 1e-4
	MaxN         int     // longest word n-gram; default 2
	CharN        int     // if >0, add character n-grams of this length; default 0
	// TypoAugment, when >0, also trains on a copy of the data in which that
	// fraction of words carries a typo.
	TypoAugment float64
}

func (o LogRegOptions) withDefaults() LogRegOptions {
	if o.Epochs <= 0 {
		o.Epochs = 20
	}
	if o.LearningRate <= 0 {
		o.LearningRate = 0.5
	}
	if o.L2 <= 0 {
		o.L2 = 1e-4
	}
	if o.MaxN <= 0 {
		o.MaxN = 2
	}
	return o
}

// LogReg is a multinomial logistic-regression classifier over binary word
// n-gram features, trained with Adagrad. Training is deterministic.
type LogReg struct {
	labels []string
	opts   LogRegOptions
	feat   map[string]int // feature -> column
	w      [][]float64    // label -> column -> weight
	bias   []float64
}

// ids maps tokens to unique known feature columns (unknown features are dropped).
func (m *LogReg) ids(tokens []string) []int {
	seen := map[int]struct{}{}
	var out []int
	for _, f := range ngramFeatures(tokens, m.opts.MaxN, m.opts.CharN) {
		if id, ok := m.feat[f]; ok {
			if _, dup := seen[id]; !dup {
				seen[id] = struct{}{}
				out = append(out, id)
			}
		}
	}
	return out
}

func (m *LogReg) probs(ids []int) []float64 {
	z := make([]float64, len(m.labels))
	maxZ := math.Inf(-1)
	for l := range m.labels {
		s := m.bias[l]
		for _, id := range ids {
			s += m.w[l][id]
		}
		z[l] = s
		maxZ = math.Max(maxZ, s)
	}
	var sum float64
	for l := range z {
		z[l] = math.Exp(z[l] - maxZ)
		sum += z[l]
	}
	for l := range z {
		z[l] /= sum
	}
	return z
}

// TrainLogReg fits a classifier. It returns nil for empty input.
func TrainLogReg(examples []Example, opts LogRegOptions) *LogReg {
	if len(examples) == 0 {
		return nil
	}
	opts = opts.withDefaults()
	if opts.TypoAugment > 0 {
		examples = withTypos(examples, opts.TypoAugment)
	}
	m := &LogReg{opts: opts, feat: map[string]int{}}
	labelSet := map[string]struct{}{}
	for _, e := range examples {
		labelSet[e.Label] = struct{}{}
		for _, f := range ngramFeatures(e.Tokens, opts.MaxN, opts.CharN) {
			if _, ok := m.feat[f]; !ok {
				m.feat[f] = len(m.feat)
			}
		}
	}
	for l := range labelSet {
		m.labels = append(m.labels, l)
	}
	sort.Strings(m.labels)
	labelIdx := make(map[string]int, len(m.labels))
	for i, l := range m.labels {
		labelIdx[l] = i
	}
	nf := len(m.feat)
	m.w = make([][]float64, len(m.labels))
	acc := make([][]float64, len(m.labels)) // Adagrad accumulators
	for l := range m.w {
		m.w[l], acc[l] = make([]float64, nf), make([]float64, nf)
	}
	m.bias = make([]float64, len(m.labels))
	biasAcc := make([]float64, len(m.labels))

	rows := make([][]int, len(examples))
	gold := make([]int, len(examples))
	for i, e := range examples {
		rows[i], gold[i] = m.ids(e.Tokens), labelIdx[e.Label]
	}
	rng := rand.New(rand.NewSource(1)) //nolint:gosec // reproducible shuffling, not security
	order := rng.Perm(len(examples))
	const eps = 1e-8
	for range opts.Epochs {
		rng.Shuffle(len(order), func(a, b int) { order[a], order[b] = order[b], order[a] })
		for _, i := range order {
			p := m.probs(rows[i])
			for l := range m.labels {
				g := p[l]
				if l == gold[i] {
					g--
				}
				biasAcc[l] += g * g
				m.bias[l] -= opts.LearningRate * g / (math.Sqrt(biasAcc[l]) + eps)
				for _, id := range rows[i] {
					gw := g + opts.L2*m.w[l][id]
					acc[l][id] += gw * gw
					m.w[l][id] -= opts.LearningRate * gw / (math.Sqrt(acc[l][id]) + eps)
				}
			}
		}
	}
	return m
}

// Predict returns the most likely label and its probability.
func (m *LogReg) Predict(tokens []string) (label string, confidence float64) {
	p := m.probs(m.ids(tokens))
	best := 0
	for l := range p {
		if p[l] > p[best] {
			best = l
		}
	}
	return m.labels[best], p[best]
}

// ngramFeatures returns word n-grams up to maxN (with sentence boundary
// markers) and, when charN>0, character n-grams of each word.
func ngramFeatures(tokens []string, maxN, charN int) []string {
	padded := make([]string, 0, len(tokens)+2)
	padded = append(padded, "<s>")
	padded = append(padded, tokens...)
	padded = append(padded, "</s>")
	var out []string
	for n := 1; n <= maxN; n++ {
		for i := 0; i+n <= len(padded); i++ {
			if n == 1 && (i == 0 || i == len(padded)-1) {
				continue // boundary markers only appear inside longer n-grams
			}
			out = append(out, join(padded[i:i+n]))
		}
	}
	if charN > 0 {
		for _, t := range tokens {
			w := "^" + t + "$"
			for i := 0; i+charN <= len(w); i++ {
				out = append(out, "c:"+w[i:i+charN])
			}
		}
	}
	return out
}

func join(ss []string) string {
	n := len(ss) - 1
	for _, s := range ss {
		n += len(s)
	}
	b := make([]byte, 0, n)
	for i, s := range ss {
		if i > 0 {
			b = append(b, ' ')
		}
		b = append(b, s...)
	}
	return string(b)
}

// withTypos returns examples followed by typo-corrupted copies of themselves.
func withTypos(examples []Example, rate float64) []Example {
	us := make([]nlu.Utterance, len(examples))
	for i, e := range examples {
		us[i] = nlu.Utterance{Tokens: e.Tokens, Intent: e.Label}
	}
	out := append([]Example(nil), examples...)
	for _, u := range nlu.AddTypos(us, rate, 3) {
		out = append(out, Example{Tokens: u.Tokens, Label: u.Intent})
	}
	return out
}
