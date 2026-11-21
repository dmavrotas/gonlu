// Quickstart: train on a handful of annotated sentences and parse new ones.
package main

import (
	"fmt"

	"github.com/dmavrotas/gonlu"
	"github.com/dmavrotas/gonlu/nlu"
)

func main() {
	data := make([]nlu.Utterance, 0, 24)
	for _, city := range []string{"rome", "paris", "berlin", "madrid", "oslo", "lima"} {
		data = append(data,
			nlu.ParseAnnotated("book a table in [city : "+city+"] for [party : two]", "book_table"),
			nlu.ParseAnnotated("reserve a table in [city : "+city+"] for [party : four] people", "book_table"),
			nlu.ParseAnnotated("what is the weather in [city : "+city+"]", "get_weather"),
			nlu.ParseAnnotated("will it rain in [city : "+city+"] tomorrow", "get_weather"),
		)
	}

	model, err := gonlu.Train(data, gonlu.Options{})
	if err != nil {
		panic(err)
	}

	for _, text := range []string{
		"Book a table in Vienna for two",
		"is it going to rain in Oslo",
		"reserve a table for four people in Lisbon",
	} {
		res := model.Parse(text)
		fmt.Printf("%q\n  intent: %s (%.0f%%)\n", text, res.Intent, 100*res.Confidence)
		for _, s := range res.Slots {
			fmt.Printf("  slot:   %s = %q\n", s.Label, s.Text)
		}
	}
}
