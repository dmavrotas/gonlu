// Package slot provides slot-filling (sequence tagging) models.
package slot

import (
	"math/rand"
	"sort"
	"unicode"

	"github.com/dmavrotas/gonlu/nlu"
)

// Options configures tagger training. The zero value matches Train's defaults.
type Options struct {
	Epochs       int  // default 10
	CharFeatures bool // add prefix/suffix (1-4) and neighbour-suffix features
	// TypoAugment, when >0, also trains on a copy of the data in which that
	// fraction of words carries a typo. It trades a little clean-text accuracy
	// on open-vocabulary data for much better robustness to misspellings.
	TypoAugment float64
	// Gazetteer adds typo-tolerant lexicon features learned from the training
	// slot values (jackknifed, so training never sees its own values).
	Gazetteer bool
	// Entries supplies external known values per slot type (for example
	// {"driver": {"Lewis Hamilton"}}). Setting it also enables the gazetteer.
	Entries map[string][]string
}

// Tagger is a greedy left-to-right averaged-perceptron BIO tagger. It is small,
// dependency-free, deterministic and trains in seconds on ATIS-sized corpora.
type Tagger struct {
	opts    Options
	lex     *lexicon // used at inference; nil when the gazetteer is off
	tags    []string
	weights map[string]map[string]float64 // feature -> tag -> averaged weight
}

type trainer struct {
	w, total map[string]map[string]float64
	stamp    map[string]map[string]int
	step     int
}

func (t *trainer) update(feat, tag string, delta float64) {
	if t.w[feat] == nil {
		t.w[feat], t.total[feat], t.stamp[feat] = map[string]float64{}, map[string]float64{}, map[string]int{}
	}
	// lazily accumulate the weight over the steps it stayed unchanged
	t.total[feat][tag] += float64(t.step-t.stamp[feat][tag]) * t.w[feat][tag]
	t.stamp[feat][tag] = t.step
	t.w[feat][tag] += delta
}

func (t *trainer) average() map[string]map[string]float64 {
	out := make(map[string]map[string]float64, len(t.w))
	for f, m := range t.w {
		for tag, w := range m {
			tot := t.total[f][tag] + float64(t.step-t.stamp[f][tag])*w
			if avg := tot / float64(t.step); avg != 0 {
				if out[f] == nil {
					out[f] = map[string]float64{}
				}
				out[f][tag] = avg
			}
		}
	}
	return out
}

func shape(w string) string {
	var hasDigit, hasAlpha bool
	for _, r := range w {
		hasDigit = hasDigit || unicode.IsDigit(r)
		hasAlpha = hasAlpha || unicode.IsLetter(r)
	}
	switch {
	case hasDigit && hasAlpha:
		return "mixed"
	case hasDigit:
		return "digit"
	}
	return "alpha"
}

func at(toks []string, i int) string {
	switch {
	case i < 0:
		return "<s>"
	case i >= len(toks):
		return "</s>"
	}
	return toks[i]
}

func suffix(w string, n int) string {
	if len(w) <= n {
		return w
	}
	return w[len(w)-n:]
}

func prefix(w string, n int) string {
	if len(w) <= n {
		return w
	}
	return w[:n]
}

// features builds the sparse feature set for position i given the previous two tags.
func features(toks []string, i int, prev, prev2 string, o Options, gaz [][]string) []string {
	w := toks[i]
	p1, p2, n1, n2 := at(toks, i-1), at(toks, i-2), at(toks, i+1), at(toks, i+2)
	f := []string{
		"b", "w=" + w, "w-1=" + p1, "w-2=" + p2, "w+1=" + n1, "w+2=" + n2,
		"p3=" + prefix(w, 3), "s3=" + suffix(w, 3), "s2=" + suffix(w, 2),
		"shape=" + shape(w), "t-1=" + prev, "t-1,t-2=" + prev + "," + prev2,
		"t-1,w=" + prev + "," + w, "w-1,w=" + p1 + "," + w, "w,w+1=" + w + "," + n1,
	}
	if o.CharFeatures {
		f = append(f, "p1="+prefix(w, 1), "p2="+prefix(w, 2), "p4="+prefix(w, 4), "s1="+suffix(w, 1), "s4="+suffix(w, 4),
			"s3-1="+suffix(p1, 3), "s3+1="+suffix(n1, 3), "p3-1="+prefix(p1, 3), "p3+1="+prefix(n1, 3))
	}
	if gaz != nil {
		for _, g := range gaz[i] {
			f = append(f, g, g+"|t-1="+prev)
		}
		for _, g := range gazAt(gaz, i-1) {
			f = append(f, "L:"+g)
		}
		for _, g := range gazAt(gaz, i+1) {
			f = append(f, "R:"+g)
		}
	}
	return f
}

func gazAt(gaz [][]string, i int) []string {
	if i < 0 || i >= len(gaz) {
		return nil
	}
	return gaz[i]
}

func score(w map[string]map[string]float64, feats, tags []string) string {
	sums := make(map[string]float64, len(tags))
	for _, f := range feats {
		for tag, v := range w[f] {
			sums[tag] += v
		}
	}
	best := tags[0]
	for _, tag := range tags { // tags are sorted: deterministic tie-breaks
		if sums[tag] > sums[best] {
			best = tag
		}
	}
	return best
}

// Train fits a tagger for the given number of epochs. Utterance order is
// shuffled each epoch with a fixed seed, so training is reproducible.
// It returns nil when there is no training data.
func Train(data []nlu.Utterance, epochs int) *Tagger {
	return TrainOptions(data, Options{Epochs: epochs})
}

// TrainOptions is Train with explicit options. It returns nil for empty data.
func TrainOptions(data []nlu.Utterance, opts Options) *Tagger {
	if len(data) == 0 {
		return nil
	}
	if opts.Epochs <= 0 {
		opts.Epochs = 10
	}
	origN := len(data)
	if opts.TypoAugment > 0 {
		data = append(append([]nlu.Utterance(nil), data...), nlu.AddTypos(data, opts.TypoAugment, 3)...)
	}
	gold := make([][]string, len(data))
	tagSet := map[string]struct{}{"O": {}}
	for i, u := range data {
		gold[i] = nlu.BIOFromSpans(len(u.Tokens), u.Slots)
		for _, g := range gold[i] {
			tagSet[g] = struct{}{}
		}
	}
	tags := make([]string, 0, len(tagSet))
	for tg := range tagSet {
		tags = append(tags, tg)
	}
	sort.Strings(tags)

	useGaz := opts.Gazetteer || len(opts.Entries) > 0
	var gaz [][][]string
	var lex *lexicon
	if useGaz {
		lex = buildLexicon(data[:origN], opts.Entries)
		gaz = jackknife(data, origN, opts.Entries)
	}

	tr := &trainer{w: map[string]map[string]float64{}, total: map[string]map[string]float64{}, stamp: map[string]map[string]int{}}
	rng := rand.New(rand.NewSource(1)) //nolint:gosec // reproducible shuffling, not security
	order := rng.Perm(len(data))
	for range opts.Epochs {
		rng.Shuffle(len(order), func(a, b int) { order[a], order[b] = order[b], order[a] })
		for _, idx := range order {
			toks := data[idx].Tokens
			prev, prev2 := "O", "O"
			for i := range toks {
				var g [][]string
				if gaz != nil {
					g = gaz[idx]
				}
				feats := features(toks, i, prev, prev2, opts, g)
				guess := score(tr.w, feats, tags)
				tr.step++
				if want := gold[idx][i]; guess != want {
					for _, f := range feats {
						tr.update(f, want, 1)
						tr.update(f, guess, -1)
					}
				}
				// teacher forcing: condition on the gold history while training
				prev, prev2 = gold[idx][i], prev
			}
		}
	}
	if tr.step == 0 {
		tr.step = 1
	}
	return &Tagger{opts: opts, lex: lex, tags: tags, weights: tr.average()}
}

// Tag labels tokens and returns the predicted slot spans.
func (t *Tagger) Tag(tokens []string) []nlu.Span {
	if len(tokens) == 0 {
		return nil
	}
	out := make([]string, len(tokens))
	var gaz [][]string
	if t.lex != nil {
		gaz = t.lex.tokenFeatures(tokens)
	}
	prev, prev2 := "O", "O"
	for i := range tokens {
		out[i] = score(t.weights, features(tokens, i, prev, prev2, t.opts, gaz), t.tags)
		prev, prev2 = out[i], prev
	}
	return nlu.SpansFromBIO(out)
}
