// SNIPS: train on the real SNIPS benchmark (run `go run ./cmd/nlubench fetch`
// first) and parse voice-assistant style requests.
package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/dmavrotas/gonlu"
	"github.com/dmavrotas/gonlu/bench/dataset"
	"github.com/dmavrotas/gonlu/nlu"
)

var intents = []string{"AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"}

func load(dir string) ([]nlu.Utterance, error) {
	var all []nlu.Utterance
	for _, in := range intents {
		f, err := os.Open(filepath.Join(dir, "snips", "train_"+in+"_full.json")) //nolint:gosec // example path
		if err != nil {
			return nil, fmt.Errorf("%w (run `go run ./cmd/nlubench fetch` first)", err)
		}
		us, err := dataset.ParseSnips(io.Reader(f))
		_ = f.Close()
		if err != nil {
			return nil, err
		}
		all = append(all, us...)
	}
	return all, nil
}

func main() {
	data, err := load("data")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	model, err := gonlu.Train(data, gonlu.Options{})
	if err != nil {
		panic(err)
	}
	fmt.Printf("trained on %d SNIPS utterances\n\n", len(data))
	for _, text := range []string{
		"Play some Beatles on Spotify",
		"What's the weather like in Paris tomorrow",
		"Book a table for 4 at a pizza place in Boston",
		"Add Bohemian Rhapsody to my road trip playlist",
		"Rate this book 5 out of 6 stars",
		"When is Black Panther playing at the Rialto cinema",
	} {
		res := model.Parse(text)
		fmt.Printf("%q\n  intent: %s (%.0f%%)\n", text, res.Intent, 100*res.Confidence)
		for _, s := range res.Slots {
			fmt.Printf("  slot:   %-24s %q\n", s.Label, s.Text)
		}
	}
}
