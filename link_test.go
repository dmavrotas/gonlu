package gonlu

import "testing"

func TestEditDistance(t *testing.T) {
	tests := []struct {
		a, b string
		want int
	}{
		{"hamilton", "hamilton", 0},
		{"hamliton", "hamilton", 1}, //nolint:misspell // transposition
		{"hamilto", "hamilton", 1},  // deletion
		{"hamiltxn", "hamilton", 1}, // substitution
		{"hamiltonn", "hamilton", 1},
		{"abc", "xyz", 3},
		{"", "abc", 3},
	}
	for _, tt := range tests {
		if got := editDistance(tt.a, tt.b); got != tt.want {
			t.Errorf("editDistance(%q,%q) = %d, want %d", tt.a, tt.b, got, tt.want)
		}
	}
}

func TestLinkerResolution(t *testing.T) {
	l := newLinker(map[string][]string{
		"driver": {"Lewis Hamilton", "Max Verstappen", "Duncan Hamilton", "Charles Leclerc"},
		"team":   {"Red Bull", "Ferrari"},
	})
	tests := []struct {
		name, label, text, want string
	}{
		{"exact full entry", "driver", "max verstappen", "Max Verstappen"},
		{"surname token", "driver", "verstappen", "Max Verstappen"},
		{"ambiguous token: earlier entry wins", "driver", "hamilton", "Lewis Hamilton"},
		{"typo in surname", "driver", "hamliton", "Lewis Hamilton"}, //nolint:misspell // intentional typo
		{"typo in full name", "driver", "charles leclrc", "Charles Leclerc"},
		{"typo in multi-word team", "team", "red bul", "Red Bull"},
		{"too short for fuzzy", "driver", "max", "Max Verstappen"}, // exact token still matches
		{"short word no fuzzy", "team", "fer", ""},
		{"unrelated", "driver", "zzzzzz", ""},
		{"unknown type", "circuit", "monza", ""},
		{"dotted label falls back to its last part", "pilot.driver", "verstappen", "Max Verstappen"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := l.link(tt.label, tt.text); got != tt.want {
				t.Errorf("link(%q, %q) = %q, want %q", tt.label, tt.text, got, tt.want)
			}
		})
	}
}

func TestLinkerNilSafe(t *testing.T) {
	if got := newLinker(nil).link("driver", "hamilton"); got != "" {
		t.Errorf("empty linker returned %q", got)
	}
}

func TestParseLinksSlotsToEntries(t *testing.T) {
	m, err := Train(trainingData(), Options{Entries: map[string][]string{"city": {"Lisbon", "Vienna", "Salzburg"}}})
	if err != nil {
		t.Fatal(err)
	}
	res := m.Parse("book a table in vienan for two") // typo
	var city *Slot
	for i := range res.Slots {
		if res.Slots[i].Label == "city" {
			city = &res.Slots[i]
		}
	}
	if city == nil {
		t.Fatalf("no city slot in %+v", res.Slots)
	}
	if city.Entry != "Vienna" {
		t.Errorf("Entry = %q, want Vienna (text %q)", city.Entry, city.Text)
	}
	// A slot type with no entries has no link.
	for _, s := range res.Slots {
		if s.Label == "party" && s.Entry != "" {
			t.Errorf("party linked to %q", s.Entry)
		}
	}
}

func TestModelLink(t *testing.T) {
	m, err := Train(trainingData(), Options{Entries: map[string][]string{
		"driver": {"Lewis Hamilton", "Max Verstappen"},
		"team":   {"Ferrari", "Red Bull"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name, typ, text string
		wantEntry       string
		wantOK          bool
		wantCost        int
	}{
		{"exact full entry", "driver", "lewis hamilton", "Lewis Hamilton", true, 0},
		{"exact word of an entry", "driver", "hamilton", "Lewis Hamilton", true, 1},
		{"typo", "team", "ferari", "Ferrari", true, 2},
		{"wrong type finds nothing", "driver", "ferari", "", false, 0},
		{"unknown type", "circuit", "monza", "", false, 0},
		{"nothing close", "driver", "zzzzzz", "", false, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			entry, cost, ok := m.Link(tt.typ, tt.text)
			if entry != tt.wantEntry || ok != tt.wantOK || (ok && cost != tt.wantCost) {
				t.Errorf("Link(%q, %q) = (%q, %d, %v), want (%q, %d, %v)", tt.typ, tt.text, entry, cost, ok, tt.wantEntry, tt.wantCost, tt.wantOK)
			}
		})
	}
}

func TestLinkCostOrdersMatchQuality(t *testing.T) {
	m, _ := Train(trainingData(), Options{Entries: map[string][]string{"driver": {"Lewis Hamilton"}, "team": {"Hamilton Racing"}}})
	_, driverCost, _ := m.Link("driver", "hamilton") // an exact word of a driver entry
	_, teamCost, _ := m.Link("team", "hamiltn")      // a typo of a team word
	if driverCost >= teamCost {
		t.Errorf("exact word (cost %d) should beat a typo (cost %d)", driverCost, teamCost)
	}
}
