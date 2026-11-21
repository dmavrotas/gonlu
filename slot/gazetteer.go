package slot

import (
	"sort"
	"strings"

	"github.com/dmavrotas/gonlu/nlu"
)

// lexicon maps words seen inside slot values to slot types, with a
// symmetric-delete index so that words within one edit (insertion, deletion,
// substitution or adjacent transposition) of a known word also match.
type lexicon struct {
	exact map[string]map[string]struct{} // word -> types
	fuzzy map[string]map[string]struct{} // word-or-one-deletion -> types
}

// slotType reduces a label like "fromloc.city_name" to "city_name" so related
// labels share vocabulary.
func slotType(label string) string {
	if i := strings.LastIndex(label, "."); i >= 0 {
		return label[i+1:]
	}
	return label
}

// minFuzzyLen is the shortest word that may match approximately; shorter words
// collide too often.
const minFuzzyLen = 4

// jackknifeFolds is the number of folds used so a training utterance never sees
// a lexicon containing its own slot values.
const jackknifeFolds = 5

func deletions(w string) []string {
	out := make([]string, 0, len(w)+1)
	out = append(out, w)
	for i := range w {
		out = append(out, w[:i]+w[i+1:])
	}
	return out
}

func add(m map[string]map[string]struct{}, k, typ string) {
	if m[k] == nil {
		m[k] = map[string]struct{}{}
	}
	m[k][typ] = struct{}{}
}

func (lx *lexicon) addWord(w, typ string) {
	add(lx.exact, w, typ)
	if len(w) >= minFuzzyLen {
		for _, k := range deletions(w) {
			add(lx.fuzzy, k, typ)
		}
	}
}

// buildLexicon indexes the slot values of data plus any external entries
// (slot type -> known values, each value tokenised like input text).
func buildLexicon(data []nlu.Utterance, entries map[string][]string) *lexicon {
	lx := &lexicon{exact: map[string]map[string]struct{}{}, fuzzy: map[string]map[string]struct{}{}}
	for _, u := range data {
		for _, s := range u.Slots {
			typ := slotType(s.Label)
			for _, w := range u.Tokens[s.Start:s.End] {
				lx.addWord(w, typ)
			}
		}
	}
	for typ, values := range entries {
		for _, v := range values {
			for _, w := range nlu.Tokenize(v) {
				lx.addWord(w, slotType(typ))
			}
		}
	}
	return lx
}

func sortedKeys(m map[string]struct{}) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// features returns "gz:<type>" for exact matches and "gzf:<type>" for matches
// that are only approximate, in a deterministic order.
func (lx *lexicon) features(w string) []string {
	var out []string
	exact := lx.exact[w]
	for _, typ := range sortedKeys(exact) {
		out = append(out, "gz:"+typ)
	}
	if len(w) >= minFuzzyLen {
		approx := map[string]struct{}{}
		for _, k := range deletions(w) {
			for typ := range lx.fuzzy[k] {
				if _, ok := exact[typ]; !ok {
					approx[typ] = struct{}{}
				}
			}
		}
		for _, typ := range sortedKeys(approx) {
			out = append(out, "gzf:"+typ)
		}
	}
	return out
}

func (lx *lexicon) tokenFeatures(tokens []string) [][]string {
	out := make([][]string, len(tokens))
	for i, w := range tokens {
		out[i] = lx.features(w)
	}
	return out
}

// jackknife returns per-utterance gazetteer features where utterance i only
// sees a lexicon built from the other folds (plus external entries). Data may
// hold n originals followed by perturbed copies; copy i%n shares its original's
// fold so a copy never sees its own original's slot values.
func jackknife(data []nlu.Utterance, n int, entries map[string][]string) [][][]string {
	foldOf := func(i int) int { return (i % n) % jackknifeFolds }
	out := make([][][]string, len(data))
	for fold := range jackknifeFolds {
		var rest []nlu.Utterance
		for i, u := range data {
			if foldOf(i) != fold {
				rest = append(rest, u)
			}
		}
		lx := buildLexicon(rest, entries)
		for i, u := range data {
			if foldOf(i) == fold {
				out[i] = lx.tokenFeatures(u.Tokens)
			}
		}
	}
	return out
}
