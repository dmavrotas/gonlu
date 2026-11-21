// Package dataset parses public intent/slot benchmark corpora into nlu.Utterance.
package dataset

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/dmavrotas/nlu/nlu"
)

// ParseATIS reads the ATIS ".w-intent.iob" format: tab-separated, tokens
// (wrapped in BOS/EOS) on the left, one IOB tag per token on the right. The tag
// aligned with EOS is the intent label.
// Multi-intent labels such as "a#b" are kept verbatim as one class.
func ParseATIS(r io.Reader) ([]nlu.Utterance, error) {
	var out []nlu.Utterance
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for n := 1; sc.Scan(); n++ {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		left, right, ok := strings.Cut(line, "\t")
		if !ok {
			return nil, fmt.Errorf("line %d: missing tab separator", n)
		}
		words, fields := strings.Fields(left), strings.Fields(right)
		if len(words) < 3 || len(fields) != len(words) {
			return nil, fmt.Errorf("line %d: %d tokens but %d tags", n, len(words), len(fields))
		}
		intent := fields[len(fields)-1]
		// Drop the BOS/EOS sentinels; the tag at EOS was the intent.
		words, tags := words[1:len(words)-1], fields[1:len(fields)-1]
		out = append(out, nlu.Utterance{Tokens: words, Intent: intent, Slots: spansFromIOB(tags)})
	}
	return out, sc.Err()
}

// spansFromIOB converts IOB tags to spans. A stray I- tag starts a new span.
func spansFromIOB(tags []string) []nlu.Span {
	var spans []nlu.Span
	start, label := -1, ""
	flush := func(end int) {
		if start >= 0 {
			spans = append(spans, nlu.Span{Start: start, End: end, Label: label})
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

type snipsFile map[string][]struct {
	Data []struct {
		Text   string `json:"text"`
		Entity string `json:"entity"`
	} `json:"data"`
}

// ParseSnips reads one Snips NLU benchmark JSON file ({"Intent": [{"data": [...]}]}).
func ParseSnips(r io.Reader) ([]nlu.Utterance, error) {
	var f snipsFile
	if err := json.NewDecoder(r).Decode(&f); err != nil {
		return nil, err
	}
	var out []nlu.Utterance
	for intent, items := range f {
		for _, item := range items {
			u := nlu.Utterance{Intent: intent}
			for _, chunk := range item.Data {
				toks := nlu.Tokenize(chunk.Text)
				if chunk.Entity != "" && len(toks) > 0 {
					u.Slots = append(u.Slots, nlu.Span{Start: len(u.Tokens), End: len(u.Tokens) + len(toks), Label: chunk.Entity})
				}
				u.Tokens = append(u.Tokens, toks...)
			}
			out = append(out, u)
		}
	}
	return out, nil
}
