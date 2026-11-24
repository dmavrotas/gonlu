package intent

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// persistCorpus is shared, sentence for sentence, with rslu's
// tests/persist.rs, so the cross-library fixtures describe the same model.
func persistCorpus() []Example {
	var data []Example
	for _, city := range []string{"rome", "paris", "berlin", "madrid", "oslo", "lima", "quito"} {
		data = append(data,
			ex("book", "book a table in "+city+" for two"),
			ex("weather", "what is the weather in "+city),
			ex("music", "play some music from "+city),
		)
	}
	return data
}

var persistProbes = [][]string{
	{"book", "a", "table", "in", "vienna"},
	{"weather", "in", "tokyo", "tomorrow"},
	{"play", "jazz"},
	{},
	{"completely", "unrelated", "words"},
	{"bokk", "a", "tabel"},
}

func persistProfiles() map[string]LogRegOptions {
	return map[string]LogRegOptions{
		"default": DefaultOptions(),
		"binary":  {},
		"robust":  RobustOptions(),
	}
}

func TestLoadedModelPredictsExactlyLikeTrainedOne(t *testing.T) {
	for name, opts := range persistProfiles() {
		m := TrainLogReg(persistCorpus(), opts)
		b, err := m.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		var loaded LogReg
		if err := loaded.UnmarshalBinary(b); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		for _, probe := range persistProbes {
			wl, wp := m.Predict(probe)
			gl, gp := loaded.Predict(probe)
			if gl != wl || math.Float64bits(gp) != math.Float64bits(wp) {
				t.Errorf("%s %v: got (%s, %v), want (%s, %v)", name, probe, gl, gp, wl, wp)
			}
		}
		again, _ := loaded.MarshalBinary()
		if !bytes.Equal(again, b) {
			t.Errorf("%s: re-encoding changed the bytes", name)
		}
	}
}

func TestWriteToAndReadLogReg(t *testing.T) {
	m := TrainLogReg(persistCorpus(), LogRegOptions{})
	var buf bytes.Buffer
	if _, err := m.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	want, _ := m.MarshalBinary()
	loaded, err := ReadLogReg(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := loaded.MarshalBinary(); !bytes.Equal(got, want) {
		t.Error("ReadLogReg(WriteTo(m)) differs from m")
	}
}

func TestForeignAndDamagedInputIsRejected(t *testing.T) {
	b, _ := TrainLogReg(persistCorpus(), LogRegOptions{}).MarshalBinary()
	var m LogReg
	if err := m.UnmarshalBinary([]byte(`{"json": true}`)); !errors.Is(err, ErrNotAModel) {
		t.Errorf("json: %v", err)
	}
	if err := m.UnmarshalBinary(append(append([]byte(nil), b...), 0)); !errors.Is(err, ErrCorrupt) {
		t.Errorf("trailing byte: %v", err)
	}
	future := append([]byte("RSLU"), 99, 0, 0, 0)
	if err := m.UnmarshalBinary(future); !errors.Is(err, ErrUnsupportedVersion) {
		t.Errorf("future version: %v", err)
	}
	for cut := range len(b) {
		if err := m.UnmarshalBinary(b[:cut]); err == nil {
			t.Fatalf("prefix of %d bytes decoded", cut)
		}
	}
}

func TestFlippedBytesNeverPanic(t *testing.T) {
	b, _ := TrainLogReg(persistCorpus()[:6], DefaultOptions()).MarshalBinary()
	for i := range b {
		damaged := append([]byte(nil), b...)
		damaged[i] ^= 0xff
		var m LogReg
		// Either outcome is fine (a flipped weight is still a valid model);
		// decoding and predicting must not panic or hang.
		if m.UnmarshalBinary(damaged) == nil {
			m.Predict([]string{"book", "a", "table"})
		}
	}
}

// Cross-library fixtures: a model saved by one library (NAME-logreg.bin) and
// the predictions that library made with it (NAME-logreg.txt), one probe per
// line: label, probability as hex float64 bits, space-separated tokens.

type probe struct {
	tokens []string
	label  string
	p      float64
}

func readProbes(t *testing.T, path string) []probe {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var out []probe
	for _, line := range strings.Split(strings.TrimRight(string(raw), "\n"), "\n") {
		parts := strings.SplitN(line, "\t", 3)
		if len(parts) != 3 {
			t.Fatalf("bad fixture line %q", line)
		}
		bits, err := strconv.ParseUint(parts[1], 16, 64)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, probe{strings.Fields(parts[2]), parts[0], math.Float64frombits(bits)})
	}
	return out
}

// TestLoadsModelsSavedByRslu checks cross-library compatibility: the model in
// testdata was trained and saved by rslu (see rslu's tests/persist.rs).
// Probabilities may differ in the last bits (the two libraries sum in a
// different order), never in the label.
func TestLoadsModelsSavedByRslu(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("testdata", "rslu-logreg.bin"))
	if err != nil {
		t.Fatal(err)
	}
	var m LogReg
	if err := m.UnmarshalBinary(b); err != nil {
		t.Fatal(err)
	}
	probes := readProbes(t, filepath.Join("testdata", "rslu-logreg.txt"))
	if len(probes) == 0 {
		t.Fatal("fixture has no probes")
	}
	for _, p := range probes {
		label, prob := m.Predict(p.tokens)
		if label != p.label || math.Abs(prob-p.p) > 1e-12 {
			t.Errorf("%v: got (%s, %.17g), rslu said (%s, %.17g)", p.tokens, label, prob, p.label, p.p)
		}
	}
}

// TestWriteFixtureForRslu regenerates the fixture rslu's tests load, when
// GONLU_WRITE_FIXTURE names the output directory.
func TestWriteFixtureForRslu(t *testing.T) {
	dir := os.Getenv("GONLU_WRITE_FIXTURE")
	if dir == "" {
		t.Skip("set GONLU_WRITE_FIXTURE=<dir> to regenerate")
	}
	m := TrainLogReg(persistCorpus(), DefaultOptions())
	b, _ := m.MarshalBinary()
	if err := os.WriteFile(filepath.Join(dir, "gonlu-logreg.bin"), b, 0o644); err != nil {
		t.Fatal(err)
	}
	var lines strings.Builder
	for _, tokens := range persistProbes {
		label, p := m.Predict(tokens)
		fmt.Fprintf(&lines, "%s\t%016x\t%s\n", label, math.Float64bits(p), strings.Join(tokens, " "))
	}
	if err := os.WriteFile(filepath.Join(dir, "gonlu-logreg.txt"), []byte(lines.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}
