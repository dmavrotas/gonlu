package nlu

import "regexp"

var annotatedSlot = regexp.MustCompile(`\[\s*([^\[\]:]+?)\s*:\s*([^\[\]]+?)\s*\]`)

// ParseAnnotated turns text with inline slot markup into a training Utterance.
// Slots are written "[label : value]", for example
//
//	ParseAnnotated("wake me up at [time : nine am]", "alarm_set")
//
// The text is tokenised with Tokenize, so spans index into the returned tokens.
func ParseAnnotated(text, intent string) Utterance {
	u := Utterance{Intent: intent}
	last := 0
	for _, m := range annotatedSlot.FindAllStringSubmatchIndex(text, -1) {
		u.Tokens = append(u.Tokens, Tokenize(text[last:m[0]])...)
		toks := Tokenize(text[m[4]:m[5]])
		if len(toks) > 0 {
			u.Slots = append(u.Slots, Span{Start: len(u.Tokens), End: len(u.Tokens) + len(toks), Label: text[m[2]:m[3]]})
		}
		u.Tokens = append(u.Tokens, toks...)
		last = m[1]
	}
	u.Tokens = append(u.Tokens, Tokenize(text[last:])...)
	return u
}
