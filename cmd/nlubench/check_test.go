package main

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestCompareDetectsRegressions(t *testing.T) {
	base := baseline{
		"ATIS":      {Intent: 93.7, Slot: 93.3, Frame: 79.6},
		"BANKING77": {Intent: 91.0},
	}
	tests := []struct {
		name string
		cur  baseline
		tol  float64
		want []string
	}{
		{"identical", base, 0.5, nil},
		{"small drop within tolerance", baseline{"ATIS": {Intent: 93.4, Slot: 93.3, Frame: 79.6}, "BANKING77": {Intent: 91.0}}, 0.5, nil},
		{"improvement is fine", baseline{"ATIS": {Intent: 95, Slot: 94, Frame: 81}, "BANKING77": {Intent: 92}}, 0.5, nil},
		{"intent regression", baseline{"ATIS": {Intent: 92.0, Slot: 93.3, Frame: 79.6}, "BANKING77": {Intent: 91.0}}, 0.5, []string{"ATIS intent"}},
		{"several metrics", baseline{"ATIS": {Intent: 93.7, Slot: 90.0, Frame: 70.0}, "BANKING77": {Intent: 85}}, 0.5,
			[]string{"ATIS slot-F1", "ATIS frame", "BANKING77 intent"}},
		{"missing benchmark is a violation", baseline{"ATIS": base["ATIS"]}, 0.5, []string{"BANKING77 missing"}},
		{"unit boundary", baseline{"ATIS": {Intent: 93.2, Slot: 93.3, Frame: 79.6}, "BANKING77": {Intent: 91.0}}, 0.5, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got []string
			for _, v := range compare(base, tt.cur, tt.tol) {
				got = append(got, v.key())
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("violations = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBaselineRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sub", "baseline.json")
	want := baseline{"ATIS": {Intent: 93.73, Slot: 93.27, Frame: 79.62}, "CLINC150": {Intent: 91.67}}
	if err := writeBaseline(path, want); err != nil {
		t.Fatal(err)
	}
	got, err := readBaseline(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("round trip: got %v, want %v", got, want)
	}
	if _, err := readBaseline(filepath.Join(t.TempDir(), "nope.json")); err == nil || !strings.Contains(err.Error(), "baseline") {
		t.Errorf("missing file error should mention the baseline, got %v", err)
	}
}
