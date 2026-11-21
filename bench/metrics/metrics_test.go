package metrics

import (
	"math"
	"testing"

	"github.com/dmavrotas/gonlu/nlu"
)

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestAccuracy(t *testing.T) {
	if got := Accuracy([]string{"a", "b", "c", "d"}, []string{"a", "b", "x", "d"}); !near(got, 0.75) {
		t.Errorf("got %v want 0.75", got)
	}
	if got := Accuracy(nil, nil); got != 0 {
		t.Errorf("empty = %v, want 0", got)
	}
	if got := Accuracy([]string{"a"}, []string{"a", "b"}); !near(got, 0.5) {
		t.Errorf("short prediction = %v, want 0.5", got)
	}
}

func TestSlotF1(t *testing.T) {
	want := [][]nlu.Span{{{Start: 0, End: 1, Label: "city"}, {Start: 3, End: 4, Label: "date"}}, {{Start: 1, End: 2, Label: "city"}}}
	got := [][]nlu.Span{{{Start: 0, End: 1, Label: "city"}, {Start: 3, End: 4, Label: "time"}}, {{Start: 1, End: 2, Label: "city"}, {Start: 0, End: 1, Label: "x"}}}
	r := SlotF1(got, want) // tp=2, predicted=4, gold=3
	if !near(r.Precision, 0.5) || !near(r.Recall, 2.0/3) {
		t.Errorf("P/R = %v/%v", r.Precision, r.Recall)
	}
	if !near(r.F1, 2*0.5*(2.0/3)/(0.5+2.0/3)) {
		t.Errorf("F1 = %v", r.F1)
	}
	if z := SlotF1(nil, nil); z.F1 != 0 {
		t.Errorf("empty F1 = %v", z.F1)
	}
}

func TestFrameAccuracy(t *testing.T) {
	wi, gi := []string{"a", "b"}, []string{"a", "b"}
	ws := [][]nlu.Span{{{Start: 0, End: 1, Label: "x"}, {Start: 2, End: 3, Label: "y"}}, {}}
	gs := [][]nlu.Span{{{Start: 2, End: 3, Label: "y"}, {Start: 0, End: 1, Label: "x"}}, {{Start: 0, End: 1, Label: "z"}}} // 1st: order-insensitive match; 2nd: extra slot
	if got := FrameAccuracy(gi, wi, gs, ws); !near(got, 0.5) {
		t.Errorf("got %v want 0.5", got)
	}
}
