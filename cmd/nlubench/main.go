// Command nlubench downloads public NLU benchmarks and scores the library on them.
//
//	nlubench fetch [-dir data]   download ATIS and SNIPS
//	nlubench run   [-dir data]   evaluate and print a report
package main

import (
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"github.com/dmavrotas/gonlu/bench/dataset"
	"github.com/dmavrotas/gonlu/bench/metrics"
	"github.com/dmavrotas/gonlu/intent"
	"github.com/dmavrotas/gonlu/nlu"
)

const (
	atisBase  = "https://raw.githubusercontent.com/yvchen/JointSLU/master/data/"
	snipsBase = "https://raw.githubusercontent.com/snipsco/nlu-benchmark/master/2017-06-custom-intent-engines/"
)

var (
	atisFiles   = []string{"atis.train.w-intent.iob", "atis.test.w-intent.iob"}
	snipsIntent = []string{"AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"}
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: nlubench fetch|run [-dir data]")
		os.Exit(2)
	}
	fs := flag.NewFlagSet(os.Args[1], flag.ExitOnError)
	dir := fs.String("dir", "data", "dataset directory")
	_ = fs.Parse(os.Args[2:])

	var err error
	switch os.Args[1] {
	case "fetch":
		err = fetch(*dir)
	case "run":
		err = run(*dir)
	default:
		err = fmt.Errorf("unknown command %q", os.Args[1])
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func download(url, dst string) error {
	if _, err := os.Stat(dst); err == nil { //nolint:gosec // dst is built from the -dir flag and fixed names
		return nil
	}
	resp, err := http.Get(url) //nolint:gosec // fixed benchmark URLs
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s: %s", url, resp.Status)
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o750); err != nil { //nolint:gosec // dst is built from the -dir flag and fixed names
		return err
	}
	f, err := os.Create(dst) //nolint:gosec // dst is built from the -dir flag and fixed names
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	_, err = io.Copy(f, resp.Body)
	return err
}

func fetch(dir string) error {
	for _, f := range atisFiles {
		if err := download(atisBase+f, filepath.Join(dir, "atis", f)); err != nil {
			return err
		}
	}
	for _, in := range snipsIntent {
		for _, f := range []string{"train_" + in + "_full.json", "validate_" + in + ".json"} {
			if err := download(snipsBase+in+"/"+f, filepath.Join(dir, "snips", f)); err != nil {
				return err
			}
		}
	}
	fmt.Println("datasets ready in", dir)
	return nil
}

func load(path string, parse func(io.Reader) ([]nlu.Utterance, error)) ([]nlu.Utterance, error) {
	f, err := os.Open(path) //nolint:gosec // path is built from the -dir flag and fixed names
	if err != nil {
		return nil, fmt.Errorf("%w (run `nlubench fetch` first)", err)
	}
	defer func() { _ = f.Close() }()
	return parse(f)
}

func loadSnips(dir, prefix, suffix string) ([]nlu.Utterance, error) {
	var all []nlu.Utterance
	for _, in := range snipsIntent {
		us, err := load(filepath.Join(dir, "snips", prefix+in+suffix), dataset.ParseSnips)
		if err != nil {
			return nil, err
		}
		all = append(all, us...)
	}
	return all, nil
}

func run(dir string) error {
	atisTrain, err := load(filepath.Join(dir, "atis", atisFiles[0]), dataset.ParseATIS)
	if err != nil {
		return err
	}
	atisTest, err := load(filepath.Join(dir, "atis", atisFiles[1]), dataset.ParseATIS)
	if err != nil {
		return err
	}
	snipsTrain, err := loadSnips(dir, "train_", "_full.json")
	if err != nil {
		return err
	}
	snipsTest, err := loadSnips(dir, "validate_", ".json")
	if err != nil {
		return err
	}
	fmt.Printf("%-8s %6s %6s %12s\n", "dataset", "train", "test", "intent-acc")
	report("ATIS", atisTrain, atisTest)
	report("SNIPS", snipsTrain, snipsTest)
	return nil
}

func report(name string, train, test []nlu.Utterance) {
	ex := make([]intent.Example, len(train))
	for i, u := range train {
		ex[i] = intent.Example{Tokens: u.Tokens, Label: u.Intent}
	}
	nb := intent.TrainNaiveBayes(ex)
	got, want := make([]string, 0, len(test)), make([]string, 0, len(test))
	for _, u := range test {
		l, _ := nb.Predict(u.Tokens)
		got, want = append(got, l), append(want, u.Intent)
	}
	fmt.Printf("%-8s %6d %6d %11.2f%%\n", name, len(train), len(test), 100*metrics.Accuracy(got, want))
}
