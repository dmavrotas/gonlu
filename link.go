package gonlu

import (
	"strings"

	"github.com/dmavrotas/gonlu/nlu"
)

// linker resolves extracted slot text to the closest known entry of the same
// slot type, tolerating typos. Entries earlier in a list win ties, so callers
// can rank entries by importance (for example drivers by career length).
type linker struct {
	byType map[string][]linkEntry
}

type linkEntry struct {
	value  string   // original spelling, as supplied
	norm   string   // lower-cased, tokenised, space-joined
	tokens []string // norm split into words
}

func newLinker(entries map[string][]string) *linker {
	l := &linker{byType: make(map[string][]linkEntry, len(entries))}
	for typ, values := range entries {
		key := lastPart(typ)
		for _, v := range values {
			toks := nlu.Tokenize(v)
			if len(toks) == 0 {
				continue
			}
			l.byType[key] = append(l.byType[key], linkEntry{value: v, norm: strings.Join(toks, " "), tokens: toks})
		}
	}
	return l
}

// lastPart reduces a dotted label like "fromloc.city_name" to "city_name".
func lastPart(label string) string {
	if i := strings.LastIndex(label, "."); i >= 0 {
		return label[i+1:]
	}
	return label
}

// maxEdits is the typo budget for a word of the given length: none for very
// short words (too many false matches), one edit up to 7 letters, two beyond.
func maxEdits(n int) int {
	switch {
	case n <= 3:
		return 0
	case n <= 7:
		return 1
	}
	return 2
}

// link returns the best entry for text, or "" when nothing is close enough.
// Preference order: exact full entry, exact word of an entry, typo-tolerant
// full entry, typo-tolerant word of an entry; ties go to the earlier entry.
func (l *linker) link(label, text string) string {
	entry, _, _ := l.best(label, text)
	return entry
}

// best is link plus the match cost (see entryCost); ok is false when nothing is
// close enough.
func (l *linker) best(label, text string) (entry string, cost int, ok bool) {
	cands := l.byType[lastPart(label)]
	if len(cands) == 0 || text == "" {
		return "", 0, false
	}
	budget := maxEdits(len(text))
	bestCost := noMatch
	for _, e := range cands {
		if c := entryCost(e, text, budget); c < bestCost {
			entry, bestCost = e.value, c
		}
	}
	if bestCost == noMatch {
		return "", 0, false
	}
	return entry, bestCost, true
}

// noMatch is the cost of an entry that is not close enough.
const noMatch = 1 << 30

// entryCost scores how well text matches an entry (lower is better, large is
// no match). Full-entry matches cost 2*d, word matches 2*d+1, so an exact full
// entry beats an exact word, which beats any one-typo match.
func entryCost(e linkEntry, text string, budget int) int {
	cost := noMatch
	if d := editDistance(text, e.norm); d <= budget {
		cost = 2 * d
	}
	for _, w := range e.tokens {
		if d := editDistance(text, w); d <= budget && 2*d+1 < cost {
			cost = 2*d + 1
		}
	}
	return cost
}

// editDistance is the optimal-string-alignment distance: insertions, deletions,
// substitutions and adjacent transpositions each cost 1.
func editDistance(a, b string) int {
	ra, rb := []rune(a), []rune(b)
	prev2 := make([]int, len(rb)+1)
	prev := make([]int, len(rb)+1)
	cur := make([]int, len(rb)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(ra); i++ {
		cur[0] = i
		for j := 1; j <= len(rb); j++ {
			cost := 1
			if ra[i-1] == rb[j-1] {
				cost = 0
			}
			cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
			if i > 1 && j > 1 && ra[i-1] == rb[j-2] && ra[i-2] == rb[j-1] {
				cur[j] = min(cur[j], prev2[j-2]+1)
			}
		}
		prev2, prev, cur = prev, cur, prev2
	}
	return prev[len(rb)]
}
