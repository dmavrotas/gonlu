package dataset

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/dmavrotas/gonlu/nlu"
)

// ParseCLINC reads CLINC150's data_full.json and returns one split
// ("train", "val", "test", or the out-of-scope "oos_*" splits) as intent-only
// utterances.
func ParseCLINC(r io.Reader, split string) ([]nlu.Utterance, error) {
	var raw map[string][][]string
	if err := json.NewDecoder(r).Decode(&raw); err != nil {
		return nil, err
	}
	rows, ok := raw[split]
	if !ok {
		return nil, fmt.Errorf("clinc: unknown split %q", split)
	}
	out := make([]nlu.Utterance, 0, len(rows))
	for i, row := range rows {
		if len(row) != 2 {
			return nil, fmt.Errorf("clinc: %s row %d has %d fields", split, i, len(row))
		}
		out = append(out, nlu.Utterance{Tokens: nlu.Tokenize(row[0]), Intent: row[1]})
	}
	return out, nil
}

// ParseBanking reads a BANKING77 CSV file (header "text,category").
func ParseBanking(r io.Reader) ([]nlu.Utterance, error) {
	rows, err := csv.NewReader(r).ReadAll()
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, nil
	}
	out := make([]nlu.Utterance, 0, len(rows)-1)
	for i, row := range rows[1:] {
		if len(row) != 2 {
			return nil, fmt.Errorf("banking: row %d has %d fields", i+1, len(row))
		}
		out = append(out, nlu.Utterance{Tokens: nlu.Tokenize(row[0]), Intent: row[1]})
	}
	return out, nil
}

// ParseMassive reads a MASSIVE JSONL file and keeps one partition
// ("train", "dev" or "test"). Slots come from the inline "[type : value]" markup.
func ParseMassive(r io.Reader, partition string) ([]nlu.Utterance, error) {
	var out []nlu.Utterance
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for n := 1; sc.Scan(); n++ {
		if strings.TrimSpace(sc.Text()) == "" {
			continue
		}
		var rec struct {
			Partition string `json:"partition"`
			Intent    string `json:"intent"`
			Annot     string `json:"annot_utt"`
		}
		if err := json.Unmarshal(sc.Bytes(), &rec); err != nil {
			return nil, fmt.Errorf("massive line %d: %w", n, err)
		}
		if rec.Partition != partition {
			continue
		}
		u := nlu.ParseAnnotated(rec.Annot, rec.Intent)
		out = append(out, u)
	}
	return out, sc.Err()
}

// ParseMITBIO reads the MIT Restaurant/Movie corpora: one "tag<TAB>word" per
// line, blank lines between sentences. These are slot-only (no intent).
func ParseMITBIO(r io.Reader) ([]nlu.Utterance, error) {
	var out []nlu.Utterance
	var toks, tags []string
	flush := func() {
		if len(toks) > 0 {
			out = append(out, nlu.Utterance{Tokens: toks, Slots: nlu.SpansFromBIO(tags)})
		}
		toks, tags = nil, nil
	}
	sc := bufio.NewScanner(r)
	for n := 1; sc.Scan(); n++ {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			flush()
			continue
		}
		tag, word, ok := strings.Cut(line, "\t")
		if !ok {
			return nil, fmt.Errorf("mit-bio line %d: expected tag<TAB>word", n)
		}
		tags, toks = append(tags, tag), append(toks, word)
	}
	flush()
	return out, sc.Err()
}
