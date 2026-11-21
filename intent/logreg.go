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

// LogRegOptions tunes logistic-regression training. The zero value is the plain
// binary word-n-gram baseline; use DefaultOptions for the recommended setup.
type LogRegOptions struct {
	Epochs       int     // default 20
	LearningRate float64 // Adagrad base rate; default 0.5
	L2           float64 // weight decay on active features; default 1e-4
	MaxN         int     // longest word n-gram; default 2
	CharN        int     // if >0, add character n-grams of this length; default 0
	// TypoAugment, when >0, also trains on a copy of the data in which that
	// fraction of words carries a typo.
	TypoAugment float64
	// TFIDF weights each feature by its inverse document frequency and
	// L2-normalises every utterance's feature vector, instead of using binary
	// features. It usually helps on many-class problems.
	TFIDF bool
	// ClassBalance is the exponent alpha of class-balanced loss weighting: an
	// example of class c is weighted (N / (K*n_c))^alpha, where N is the number
	// of examples, K the number of classes and n_c the size of class c. 0 turns
	// it off, 1 is fully balanced. It counters a dominant class swallowing rare
	// ones on imbalanced data (as in ATIS).
	ClassBalance float64
	// NormPower controls how strongly TF-IDF rows are length-normalised: each
	// row is divided by ||x||^NormPower. 0 selects the default of 1 (full L2
	// normalisation); a negative value disables normalisation (IDF weights
	// only). Smaller powers keep a decisive word from being diluted by a long
	// utterance. It has no effect unless TFIDF is set.
	NormPower float64
}

// DefaultOptions returns the recommended configuration: TF-IDF weighting with
// word uni/bigrams plus character trigrams, learning rate 0.25 and
// square-root class balancing.
//
// Against the zero-value LogRegOptions (binary word features) it is better on
// the dev splits of ATIS, MASSIVE, CLINC150 and BANKING77 (up to +8.5 points) and
// within 0.2 on SNIPS, but it is 1.3 points worse on the ATIS test set, where
// the test distribution differs from train and a dominant class swallows rare
// intents. See the README's method notes.
func DefaultOptions() LogRegOptions {
	return LogRegOptions{TFIDF: true, LearningRate: 0.25, CharN: 3, ClassBalance: 0.5}
}

// RobustOptions is DefaultOptions plus typo-augmented training data, for
// noisy, human-typed input.
func RobustOptions() LogRegOptions {
	o := DefaultOptions()
	o.TypoAugment = 0.3
	return o
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
	idf    []float64      // per-column idf weight (nil unless TFIDF)
	w      [][]float64    // label -> column -> weight
	bias   []float64
}

// row is a sparse feature vector: columns and their values.
type row struct {
	ids  []int
	vals []float64
}

// row maps tokens to unique known feature columns (unknown features are
// dropped). With TFIDF the values are idf weights, L2-normalised; otherwise 1.
func (m *LogReg) row(tokens []string) row {
	seen := map[int]struct{}{}
	var r row
	for _, f := range ngramFeatures(tokens, m.opts.MaxN, m.opts.CharN) {
		if id, ok := m.feat[f]; ok {
			if _, dup := seen[id]; !dup {
				seen[id] = struct{}{}
				r.ids = append(r.ids, id)
			}
		}
	}
	r.vals = make([]float64, len(r.ids))
	var norm float64
	for i, id := range r.ids {
		v := 1.0
		if m.idf != nil {
			v = m.idf[id]
		}
		r.vals[i] = v
		norm += v * v
	}
	if m.idf != nil && norm > 0 && m.opts.NormPower >= 0 {
		pow := m.opts.NormPower
		if pow == 0 {
			pow = 1
		}
		div := math.Pow(math.Sqrt(norm), pow)
		for i := range r.vals {
			r.vals[i] /= div
		}
	}
	return r
}

func (m *LogReg) probs(r row) []float64 {
	z := make([]float64, len(m.labels))
	maxZ := math.Inf(-1)
	for l := range m.labels {
		s := m.bias[l]
		for k, id := range r.ids {
			s += m.w[l][id] * r.vals[k]
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

	if opts.TFIDF {
		m.idf = idfWeights(examples, m)
	}
	rows := make([]row, len(examples))
	gold := make([]int, len(examples))
	for i, e := range examples {
		rows[i], gold[i] = m.row(e.Tokens), labelIdx[e.Label]
	}
	classWeight := make([]float64, len(m.labels))
	for l := range classWeight {
		classWeight[l] = 1
	}
	if opts.ClassBalance > 0 {
		counts := make([]float64, len(m.labels))
		for _, g := range gold {
			counts[g]++
		}
		for l, c := range counts {
			classWeight[l] = math.Pow(float64(len(examples))/(float64(len(m.labels))*c), opts.ClassBalance)
		}
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
				g *= classWeight[gold[i]]
				biasAcc[l] += g * g
				m.bias[l] -= opts.LearningRate * g / (math.Sqrt(biasAcc[l]) + eps)
				for k, id := range rows[i].ids {
					gw := g*rows[i].vals[k] + opts.L2*m.w[l][id]
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
	p := m.probs(m.row(tokens))
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

// idfWeights returns smoothed inverse document frequencies per feature column.
func idfWeights(examples []Example, m *LogReg) []float64 {
	df := make([]float64, len(m.feat))
	for _, e := range examples {
		seen := map[int]struct{}{}
		for _, f := range ngramFeatures(e.Tokens, m.opts.MaxN, m.opts.CharN) {
			if id, ok := m.feat[f]; ok {
				if _, dup := seen[id]; !dup {
					seen[id] = struct{}{}
					df[id]++
				}
			}
		}
	}
	n := float64(len(examples))
	idf := make([]float64, len(df))
	for i, d := range df {
		idf[i] = math.Log((1+n)/(1+d)) + 1
	}
	return idf
}
