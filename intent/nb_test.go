package intent

import (
	"testing"

	"github.com/dmavrotas/gonlu/nlu"
)

func ex(label, text string) Example { return Example{Tokens: nlu.Tokenize(text), Label: label} }

func TestNaiveBayes(t *testing.T) {
	nb := TrainNaiveBayes([]Example{
		ex("weather", "what is the weather in paris"),
		ex("weather", "will it rain tomorrow"),
		ex("weather", "weather forecast for friday"),
		ex("music", "play some jazz"),
		ex("music", "play the beatles"),
		ex("music", "put on a playlist"),
	})
	tests := []struct{ in, want string }{
		{"weather in london", "weather"},
		{"play some rock", "music"},
		{"is it going to rain", "weather"},
	}
	for _, tt := range tests {
		got, conf := nb.Predict(nlu.Tokenize(tt.in))
		if got != tt.want {
			t.Errorf("Predict(%q) = %q, want %q", tt.in, got, tt.want)
		}
		if conf <= 0.5 || conf > 1 {
			t.Errorf("Predict(%q) confidence %v out of range", tt.in, conf)
		}
	}
}

func TestTrainNaiveBayesEmpty(t *testing.T) {
	if TrainNaiveBayes(nil) != nil {
		t.Error("expected nil for empty training set")
	}
}
