package intent

import (
	"testing"

	"github.com/dmavrotas/gonlu/nlu"
)

func smallSet() []Example {
	return []Example{
		ex("weather", "what is the weather in paris"),
		ex("weather", "will it rain tomorrow"),
		ex("weather", "weather forecast for friday"),
		ex("weather", "is it going to be sunny"),
		ex("music", "play some jazz"),
		ex("music", "play the beatles"),
		ex("music", "put on a playlist"),
		ex("music", "play my favourite song"),
	}
}

func TestLogRegLearnsTrainingSet(t *testing.T) {
	data := smallSet()
	m := TrainLogReg(data, LogRegOptions{})
	for _, e := range data {
		if got, _ := m.Predict(e.Tokens); got != e.Label {
			t.Errorf("Predict(%v) = %q, want %q", e.Tokens, got, e.Label)
		}
	}
}

func TestLogRegGeneralises(t *testing.T) {
	m := TrainLogReg(smallSet(), LogRegOptions{})
	tests := []struct{ in, want string }{
		{"weather in london", "weather"},
		{"play some rock", "music"},
	}
	for _, tt := range tests {
		got, conf := m.Predict(nlu.Tokenize(tt.in))
		if got != tt.want {
			t.Errorf("Predict(%q) = %q, want %q", tt.in, got, tt.want)
		}
		if conf <= 0.5 || conf > 1 {
			t.Errorf("confidence %v out of range for %q", conf, tt.in)
		}
	}
}

func TestLogRegDeterministicAndRobust(t *testing.T) {
	a := TrainLogReg(smallSet(), LogRegOptions{})
	b := TrainLogReg(smallSet(), LogRegOptions{})
	la, ca := a.Predict(nlu.Tokenize("play rain"))
	lb, cb := b.Predict(nlu.Tokenize("play rain"))
	if la != lb || ca != cb {
		t.Errorf("not deterministic: (%s,%v) vs (%s,%v)", la, ca, lb, cb)
	}
	if got, _ := a.Predict(nil); got == "" {
		t.Error("empty input should still return a label")
	}
	if got, _ := a.Predict(nlu.Tokenize("zzzz qqqq")); got == "" {
		t.Error("unseen words should still return a label")
	}
	if TrainLogReg(nil, LogRegOptions{}) != nil {
		t.Error("expected nil for empty training set")
	}
}

func TestClassifierInterface(t *testing.T) {
	var _ Classifier = (*NaiveBayes)(nil)
	var _ Classifier = (*LogReg)(nil)
}

func TestNgramFeatures(t *testing.T) {
	got := ngramFeatures([]string{"a", "bc"}, 2, 0)
	want := []string{"a", "bc", "<s> a", "a bc", "bc </s>"}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("feature %d = %q, want %q", i, got[i], want[i])
		}
	}
	chars := ngramFeatures([]string{"ab"}, 1, 3) // "^ab$" -> ^ab, ab$
	if len(chars) != 3 || chars[1] != "c:^ab" || chars[2] != "c:ab$" {
		t.Errorf("char features = %v", chars)
	}
}
