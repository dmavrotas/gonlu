package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/dmavrotas/gonlu/bench/dataset"
	"github.com/dmavrotas/gonlu/nlu"
)

// file is one downloadable benchmark file.
type file struct{ url, dst string }

// benchmark describes a public dataset: what it measures, where its files come
// from, and how to load its train and test splits.
type benchmark struct {
	name    string
	intents bool // has intent labels
	slots   bool // has slot labels
	files   []file
	load    func(dir string) (train, test []nlu.Utterance, err error)
}

type options struct {
	dir  string
	only []string
	typo float64
}

func parseFlags(args []string) (options, error) {
	o := options{dir: "data"}
	for i := 0; i < len(args); i++ {
		next := func() (string, error) {
			if i+1 >= len(args) {
				return "", fmt.Errorf("%s needs a value", args[i])
			}
			i++
			return args[i], nil
		}
		var v string
		var err error
		switch args[i] {
		case "-dir", "--dir":
			o.dir, err = next()
		case "-only", "--only":
			if v, err = next(); err == nil {
				o.only = strings.Split(v, ",")
			}
		case "-typo", "--typo":
			if v, err = next(); err == nil {
				_, err = fmt.Sscanf(v, "%g", &o.typo)
			}
		default:
			err = fmt.Errorf("unknown flag %q", args[i])
		}
		if err != nil {
			return o, err
		}
	}
	return o, nil
}

func selected(only []string) []benchmark {
	if len(only) == 0 {
		return benchmarks
	}
	var out []benchmark
	for _, b := range benchmarks {
		for _, n := range only {
			if strings.EqualFold(n, b.name) {
				out = append(out, b)
			}
		}
	}
	return out
}

func loadFile(path string, parse func(io.Reader) ([]nlu.Utterance, error)) ([]nlu.Utterance, error) {
	f, err := os.Open(path) //nolint:gosec // path is built from the -dir flag and fixed names
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	return parse(f)
}

// loadPair loads train and test files with the same parser.
func loadPair(dir, trainFile, testFile string, parse func(io.Reader) ([]nlu.Utterance, error)) (train, test []nlu.Utterance, err error) {
	if train, err = loadFile(filepath.Join(dir, trainFile), parse); err != nil {
		return nil, nil, err
	}
	test, err = loadFile(filepath.Join(dir, testFile), parse)
	return train, test, err
}

const (
	atisBase    = "https://raw.githubusercontent.com/yvchen/JointSLU/master/data/"
	snipsBase   = "https://raw.githubusercontent.com/snipsco/nlu-benchmark/master/2017-06-custom-intent-engines/"
	clincURL    = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
	bankingBase = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
	mitBase     = "https://groups.csail.mit.edu/sls/downloads/restaurant/"
	massiveURL  = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"
)

var snipsIntents = []string{"AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"}

func snipsFiles() []file {
	var fs []file
	for _, in := range snipsIntents {
		for _, f := range []string{"train_" + in + "_full.json", "validate_" + in + ".json"} {
			fs = append(fs, file{snipsBase + in + "/" + f, filepath.Join("snips", f)})
		}
	}
	return fs
}

func loadSnips(dir, prefix, suffix string) ([]nlu.Utterance, error) {
	var all []nlu.Utterance
	for _, in := range snipsIntents {
		us, err := loadFile(filepath.Join(dir, "snips", prefix+in+suffix), dataset.ParseSnips)
		if err != nil {
			return nil, err
		}
		all = append(all, us...)
	}
	return all, nil
}

var benchmarks = []benchmark{
	{
		name: "ATIS", intents: true, slots: true,
		files: []file{
			{atisBase + "atis.train.w-intent.iob", "atis/atis.train.w-intent.iob"},
			{atisBase + "atis.test.w-intent.iob", "atis/atis.test.w-intent.iob"},
		},
		load: func(dir string) ([]nlu.Utterance, []nlu.Utterance, error) {
			return loadPair(dir, "atis/atis.train.w-intent.iob", "atis/atis.test.w-intent.iob", dataset.ParseATIS)
		},
	},
	{
		name: "SNIPS", intents: true, slots: true,
		files: snipsFiles(),
		load: func(dir string) ([]nlu.Utterance, []nlu.Utterance, error) {
			train, err := loadSnips(dir, "train_", "_full.json")
			if err != nil {
				return nil, nil, err
			}
			test, err := loadSnips(dir, "validate_", ".json")
			return train, test, err
		},
	},
	{
		name: "MASSIVE", intents: true, slots: true, // fetched by fetchMassive (tar.gz)
		load: func(dir string) ([]nlu.Utterance, []nlu.Utterance, error) {
			p := filepath.Join(dir, "massive", "en-US.jsonl")
			train, err := loadFile(p, func(r io.Reader) ([]nlu.Utterance, error) { return dataset.ParseMassive(r, "train") })
			if err != nil {
				return nil, nil, err
			}
			test, err := loadFile(p, func(r io.Reader) ([]nlu.Utterance, error) { return dataset.ParseMassive(r, "test") })
			return train, test, err
		},
	},
	{
		name: "CLINC150", intents: true,
		files: []file{{clincURL, "clinc/data_full.json"}},
		load: func(dir string) ([]nlu.Utterance, []nlu.Utterance, error) {
			p := filepath.Join(dir, "clinc", "data_full.json")
			train, err := loadFile(p, func(r io.Reader) ([]nlu.Utterance, error) { return dataset.ParseCLINC(r, "train") })
			if err != nil {
				return nil, nil, err
			}
			test, err := loadFile(p, func(r io.Reader) ([]nlu.Utterance, error) { return dataset.ParseCLINC(r, "test") })
			return train, test, err
		},
	},
	{
		name: "BANKING77", intents: true,
		files: []file{{bankingBase + "train.csv", "banking77/train.csv"}, {bankingBase + "test.csv", "banking77/test.csv"}},
		load: func(dir string) ([]nlu.Utterance, []nlu.Utterance, error) {
			return loadPair(dir, "banking77/train.csv", "banking77/test.csv", dataset.ParseBanking)
		},
	},
	{
		name: "MIT-Rest", slots: true,
		files: []file{{mitBase + "restauranttrain.bio", "mit/restauranttrain.bio"}, {mitBase + "restauranttest.bio", "mit/restauranttest.bio"}},
		load: func(dir string) ([]nlu.Utterance, []nlu.Utterance, error) {
			return loadPair(dir, "mit/restauranttrain.bio", "mit/restauranttest.bio", dataset.ParseMITBIO)
		},
	},
}
