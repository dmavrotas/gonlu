// Entities: teach the model a domain vocabulary it never saw in training, see it
// survive typos, and link each extracted name to its canonical entry. This is how a query engine can plug in its own
// driver/team/circuit names.
package main

import (
	"fmt"

	"github.com/dmavrotas/gonlu"
	"github.com/dmavrotas/gonlu/nlu"
)

func main() {
	data := make([]nlu.Utterance, 0, 24)
	for _, d := range []string{"senna", "prost", "alonso", "vettel", "raikkonen", "button"} {
		data = append(data,
			nlu.ParseAnnotated("how many wins does [driver : "+d+"] have", "wins"),
			nlu.ParseAnnotated("how many races did [driver : "+d+"] win", "wins"),
			nlu.ParseAnnotated("how many poles did [driver : "+d+"] take", "poles"),
			nlu.ParseAnnotated("poles for [driver : "+d+"]", "poles"),
		)
	}

	// Names that appear nowhere in the training data:
	entries := map[string][]string{
		"driver": {"Lewis Hamilton", "Max Verstappen", "Charles Leclerc", "Fernando Alonso"},
	}
	model, err := gonlu.Train(data, gonlu.Options{Robust: true, Entries: entries})
	if err != nil {
		panic(err)
	}

	for _, text := range []string{
		"how many wins does verstappen have",
		"how many poles did hamilton take",
		"how many poles did hamliton take", //nolint:misspell // intentional typo
		"how many wnis does leclerc have",  // typo in a non-entity word
	} {
		res := model.Parse(text)
		fmt.Printf("%-40q -> %-6s", text, res.Intent)
		for _, s := range res.Slots {
			fmt.Printf("  %s=%q", s.Label, s.Text)
			if s.Entry != "" {
				fmt.Printf(" -> %q", s.Entry) // linked to the canonical entry
			}
		}
		fmt.Println()
	}
}
