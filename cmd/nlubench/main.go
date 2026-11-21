// Command nlubench downloads public NLU benchmarks and scores the library on them.
//
//	nlubench fetch [-dir data]                         download all benchmarks
//	nlubench run   [-dir data] [-only a,b] [-typo 0.3]  evaluate and print a report
//	nlubench check [-dir data] [-baseline f] [-tolerance 0.5] [-update]
//	                                                    fail if the default profile regressed
package main

import (
	"fmt"
	"os"

	"github.com/dmavrotas/gonlu/bench/metrics"
	"github.com/dmavrotas/gonlu/intent"
	"github.com/dmavrotas/gonlu/nlu"
	"github.com/dmavrotas/gonlu/slot"
)

const slotEpochs = 10

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: nlubench fetch|run [-dir data] [-only names] [-typo rate]")
		os.Exit(2)
	}
	opts, err := parseFlags(os.Args[2:])
	if err == nil {
		switch os.Args[1] {
		case "fetch":
			err = fetch(&opts)
		case "run":
			err = run(&opts)
		case "check":
			err = check(&opts)
		default:
			err = fmt.Errorf("unknown command %q", os.Args[1])
		}
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// profile bundles the intent and slot settings evaluated together.
type profile struct {
	name   string
	intent intent.LogRegOptions
	slot   slot.Options
}

var profiles = []profile{
	{"default", intent.DefaultOptions(), slot.Options{Epochs: slotEpochs}},
	{"robust", intent.RobustOptions(),
		slot.Options{Epochs: slotEpochs, CharFeatures: true, TypoAugment: 0.3}},
	{"robust+gaz", intent.RobustOptions(),
		slot.Options{Epochs: slotEpochs, CharFeatures: true, TypoAugment: 0.3, Gazetteer: true}},
}

func run(o *options) error {
	fmt.Printf("%-10s %-10s %6s %6s %11s %8s %9s\n", "dataset", "profile", "train", "test", "intent-acc", "slot-F1", "frame-acc")
	bs := selected(o.only)
	for i := range bs {
		b := &bs[i]
		train, test, err := b.load(o.dir)
		if err != nil {
			return fmt.Errorf("%s: %w (run `nlubench fetch` first)", b.name, err)
		}
		if o.typo > 0 {
			test = nlu.AddTypos(test, o.typo, 7)
		}
		for i := range profiles {
			p := &profiles[i]
			if p.slot.Gazetteer && !b.slots {
				continue // gazetteer features only affect slot filling
			}
			report(b, p, train, test)
		}
	}
	return nil
}

// scores are the metrics of one benchmark/profile, in percent. A zero value
// means the metric does not apply (for example no slots in CLINC150).
type scores struct {
	Intent float64 `json:"intent,omitempty"`
	Slot   float64 `json:"slot_f1,omitempty"`
	Frame  float64 `json:"frame,omitempty"`
}

// evaluate trains profile p on train and scores it on test.
func evaluate(b *benchmark, p *profile, train, test []nlu.Utterance) scores {
	var gotI, wantI []string
	var gotS, wantS [][]nlu.Span
	var clf intent.Classifier
	var tagger *slot.Tagger
	if b.intents {
		ex := make([]intent.Example, len(train))
		for i, u := range train {
			ex[i] = intent.Example{Tokens: u.Tokens, Label: u.Intent}
		}
		clf = intent.TrainLogReg(ex, p.intent)
	}
	if b.slots {
		tagger = slot.TrainOptions(train, p.slot)
	}
	for _, u := range test {
		if clf != nil {
			l, _ := clf.Predict(u.Tokens)
			gotI, wantI = append(gotI, l), append(wantI, u.Intent)
		}
		if tagger != nil {
			gotS, wantS = append(gotS, tagger.Tag(u.Tokens)), append(wantS, u.Slots)
		}
	}
	var sc scores
	if b.intents {
		sc.Intent = 100 * metrics.Accuracy(gotI, wantI)
	}
	if b.slots {
		sc.Slot = 100 * metrics.SlotF1(gotS, wantS).F1
	}
	if b.intents && b.slots {
		sc.Frame = 100 * metrics.FrameAccuracy(gotI, wantI, gotS, wantS)
	}
	return sc
}

func report(b *benchmark, p *profile, train, test []nlu.Utterance) {
	sc := evaluate(b, p, train, test)
	cell := func(ok bool, v float64) string {
		if !ok {
			return "-"
		}
		return fmt.Sprintf("%.2f%%", v)
	}
	fmt.Printf("%-10s %-10s %6d %6d %11s %8s %9s\n", b.name, p.name, len(train), len(test),
		cell(b.intents, sc.Intent), cell(b.slots, sc.Slot), cell(b.intents && b.slots, sc.Frame))
}
