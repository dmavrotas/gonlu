package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// baseline maps a benchmark name to the scores of the default profile.
type baseline map[string]scores

// checkSet is the benchmark subset used as a CI gate: small enough to run in a
// few minutes, broad enough to cover joint, slot-heavy and many-class tasks.
var checkSet = []string{"ATIS", "SNIPS", "CLINC150", "BANKING77"}

type violation struct {
	bench, metric string
	base, cur     float64
}

func (v violation) key() string { return v.bench + " " + v.metric }

func (v violation) String() string {
	if v.metric == "missing" {
		return v.bench + ": no result produced"
	}
	return fmt.Sprintf("%s %s: %.2f -> %.2f (%.2f points)", v.bench, v.metric, v.base, v.cur, v.cur-v.base)
}

// compare returns every metric in cur that fell more than tol points below base,
// plus benchmarks that have a baseline but no result. Improvements never fail.
func compare(base, cur baseline, tol float64) []violation {
	names := make([]string, 0, len(base))
	for n := range base {
		names = append(names, n)
	}
	sort.Strings(names)
	var out []violation
	for _, n := range names {
		c, ok := cur[n]
		if !ok {
			out = append(out, violation{bench: n, metric: "missing"})
			continue
		}
		b := base[n]
		for _, m := range []struct {
			name      string
			base, cur float64
		}{{"intent", b.Intent, c.Intent}, {"slot-F1", b.Slot, c.Slot}, {"frame", b.Frame, c.Frame}} {
			if m.base > 0 && m.cur < m.base-tol-1e-9 {
				out = append(out, violation{n, m.name, m.base, m.cur})
			}
		}
	}
	return out
}

func readBaseline(path string) (baseline, error) {
	b, err := os.ReadFile(path) //nolint:gosec // path comes from the -baseline flag
	if err != nil {
		return nil, fmt.Errorf("reading baseline %s: %w", path, err)
	}
	var out baseline
	if err := json.Unmarshal(b, &out); err != nil {
		return nil, fmt.Errorf("parsing baseline %s: %w", path, err)
	}
	return out, nil
}

func writeBaseline(path string, b baseline) error {
	data, err := json.MarshalIndent(b, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o600) //nolint:gosec // path comes from the -baseline flag
}

// check scores the default profile on checkSet and compares with the baseline,
// or rewrites the baseline with -update.
func check(o *options) error {
	only := o.only
	if len(only) == 0 {
		only = checkSet
	}
	def := &profiles[0]
	cur := baseline{}
	bs := selected(only)
	for i := range bs {
		b := &bs[i]
		train, test, err := b.load(o.dir)
		if err != nil {
			return fmt.Errorf("%s: %w (run `nlubench fetch -only %s` first)", b.name, err, b.name)
		}
		cur[b.name] = evaluate(b, def, train, test)
		fmt.Printf("%-10s intent=%.2f slot-F1=%.2f frame=%.2f\n", b.name, cur[b.name].Intent, cur[b.name].Slot, cur[b.name].Frame)
	}
	if o.update {
		if err := writeBaseline(o.baseline, cur); err != nil {
			return err
		}
		fmt.Println("baseline written to", o.baseline)
		return nil
	}
	base, err := readBaseline(o.baseline)
	if err != nil {
		return err
	}
	// Only gate on what was asked for.
	for n := range base {
		if _, ok := cur[n]; !ok && !contains(only, n) {
			delete(base, n)
		}
	}
	vs := compare(base, cur, o.tolerance)
	for _, v := range vs {
		fmt.Println("REGRESSION:", v)
	}
	if len(vs) > 0 {
		return fmt.Errorf("%d metric(s) regressed by more than %.2f points", len(vs), o.tolerance)
	}
	fmt.Printf("OK: no metric dropped more than %.2f points below the baseline\n", o.tolerance)
	return nil
}

func contains(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}
