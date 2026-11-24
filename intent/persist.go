package intent

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
)

// The binary model format (version 1) is shared with rslu, the Rust port: a
// model saved by either library loads in the other. Numbers are little-endian,
// floats are full float64 bit patterns (a loaded model predicts exactly what
// the saved one did), strings and lists carry a uint32 length.
//
//	magic "RSLU", uint32 version, string kind ("logreg")
//	options: uint64 epochs, f64 learning rate, f64 l2, uint64 max n,
//	         uint64 char n, f64 typo augment, uint8 tfidf, f64 class balance,
//	         f64 norm power
//	labels:   uint32 count, strings (sorted, distinct)
//	features: uint32 count, strings in column order
//	idf:      uint8 flag, then one f64 per feature when set (iff tfidf)
//	bias:     one f64 per label
//	weights:  one row per label, one f64 per feature
//
// Nothing may follow the weights.
const (
	formatVersion = 1
	logRegKind    = "logreg"
	// maxNgram bounds decoded n-gram lengths: prediction cost grows with them,
	// so a corrupt file must not be able to make Predict loop for ages.
	maxNgram = 16
)

var magic = []byte("RSLU")

// Errors returned when a model cannot be decoded. Wrapped errors add detail;
// test with errors.Is.
var (
	ErrNotAModel          = errors.New("gonlu: not an rslu/gonlu model")
	ErrUnsupportedVersion = errors.New("gonlu: unsupported model format version")
	ErrWrongKind          = errors.New("gonlu: the input holds another kind of model")
	ErrTruncated          = errors.New("gonlu: the model data ends early")
	ErrCorrupt            = errors.New("gonlu: corrupt model")
)

// MarshalBinary encodes the trained model (encoding.BinaryMarshaler). It
// fails only if the model has more than 2^32-1 labels or features.
func (m *LogReg) MarshalBinary() ([]byte, error) {
	if len(m.labels) > math.MaxUint32 || len(m.feat) > math.MaxUint32 {
		return nil, errors.New("gonlu: model too large to encode")
	}
	b := append([]byte(nil), magic...)
	b = binary.LittleEndian.AppendUint32(b, formatVersion)
	b = appendString(b, logRegKind)

	o := m.opts
	b = binary.LittleEndian.AppendUint64(b, uint64(o.Epochs))
	b = appendFloat(b, o.LearningRate)
	b = appendFloat(b, o.L2)
	b = binary.LittleEndian.AppendUint64(b, uint64(o.MaxN))
	b = binary.LittleEndian.AppendUint64(b, uint64(o.CharN))
	b = appendFloat(b, o.TypoAugment)
	b = append(b, boolByte(o.TFIDF))
	b = appendFloat(b, o.ClassBalance)
	b = appendFloat(b, o.NormPower)

	b = binary.LittleEndian.AppendUint32(b, uint32(len(m.labels)))
	for _, l := range m.labels {
		b = appendString(b, l)
	}
	names := make([]string, len(m.feat))
	for name, id := range m.feat {
		names[id] = name
	}
	b = binary.LittleEndian.AppendUint32(b, uint32(len(names)))
	for _, name := range names {
		b = appendString(b, name)
	}
	b = append(b, boolByte(m.idf != nil))
	for _, v := range m.idf {
		b = appendFloat(b, v)
	}
	for _, v := range m.bias {
		b = appendFloat(b, v)
	}
	for _, row := range m.w {
		for _, v := range row {
			b = appendFloat(b, v)
		}
	}
	return b, nil
}

// UnmarshalBinary decodes a model written by MarshalBinary or by rslu
// (encoding.BinaryUnmarshaler). It never panics on bad input: truncated,
// foreign or corrupt data is an error, and m is left unchanged.
func (m *LogReg) UnmarshalBinary(data []byte) error {
	r, err := newReader(data, logRegKind)
	if err != nil {
		return err
	}
	var o LogRegOptions
	o.Epochs = r.int()
	o.LearningRate = r.float()
	o.L2 = r.float()
	o.MaxN = r.int()
	o.CharN = r.int()
	o.TypoAugment = r.float()
	o.TFIDF = r.bool()
	o.ClassBalance = r.float()
	o.NormPower = r.float()
	if r.err == nil && (o.MaxN < 1 || o.MaxN > maxNgram || o.CharN < 0 || o.CharN > maxNgram) {
		r.fail(fmt.Errorf("%w: n-gram length out of range", ErrCorrupt))
	}

	nLabels := r.len(4)
	labels := make([]string, 0, nLabels)
	for range nLabels {
		labels = append(labels, r.string())
	}
	if r.err == nil && (len(labels) == 0 || !strictlySorted(labels)) {
		r.fail(fmt.Errorf("%w: labels must be non-empty, sorted and distinct", ErrCorrupt))
	}
	nFeat := r.len(4)
	feat := make(map[string]int, nFeat)
	for id := range nFeat {
		name := r.string()
		if _, dup := feat[name]; dup && r.err == nil {
			r.fail(fmt.Errorf("%w: duplicate feature", ErrCorrupt))
		}
		feat[name] = id
	}
	var idf []float64
	if r.bool() {
		idf = r.floats(nFeat)
	}
	if r.err == nil && (idf != nil) != o.TFIDF {
		r.fail(fmt.Errorf("%w: idf weights do not match the tfidf option", ErrCorrupt))
	}
	bias := r.floats(nLabels)
	w := make([][]float64, 0, nLabels)
	for range nLabels {
		w = append(w, r.floats(nFeat))
	}
	if r.err == nil && len(r.b) > 0 {
		r.fail(fmt.Errorf("%w: trailing bytes", ErrCorrupt))
	}
	if r.err != nil {
		return r.err
	}
	*m = LogReg{labels: labels, opts: o, feat: feat, idf: idf, w: w, bias: bias}
	return nil
}

// WriteTo writes MarshalBinary's encoding to w (io.WriterTo).
func (m *LogReg) WriteTo(w io.Writer) (int64, error) {
	b, err := m.MarshalBinary()
	if err != nil {
		return 0, err
	}
	n, err := w.Write(b)
	return int64(n), err
}

// ReadLogReg reads all of r and decodes it with UnmarshalBinary.
func ReadLogReg(r io.Reader) (*LogReg, error) {
	b, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("gonlu: reading the model: %w", err)
	}
	m := new(LogReg)
	if err := m.UnmarshalBinary(b); err != nil {
		return nil, err
	}
	return m, nil
}

func appendString(b []byte, s string) []byte {
	b = binary.LittleEndian.AppendUint32(b, uint32(len(s)))
	return append(b, s...)
}

func appendFloat(b []byte, v float64) []byte {
	return binary.LittleEndian.AppendUint64(b, math.Float64bits(v))
}

func boolByte(v bool) byte {
	if v {
		return 1
	}
	return 0
}

func strictlySorted(s []string) bool {
	return slices.IsSortedFunc(s, func(a, b string) int {
		if a < b {
			return -1
		}
		return 1 // equal neighbours are not strictly sorted
	})
}

// reader decodes from a byte slice, checking every bound. The first error
// sticks: later reads return zero values, so decoding code reads straight
// through and checks r.err once.
type reader struct {
	b   []byte
	err error
}

func newReader(data []byte, kind string) (*reader, error) {
	if len(data) < len(magic) || string(data[:len(magic)]) != string(magic) {
		return nil, ErrNotAModel
	}
	r := &reader{b: data[len(magic):]}
	if v := r.uint32(); r.err == nil && v != formatVersion {
		return nil, fmt.Errorf("%w: %d (this build reads %d)", ErrUnsupportedVersion, v, formatVersion)
	}
	if found := r.string(); r.err == nil && found != kind {
		return nil, fmt.Errorf("%w: %q", ErrWrongKind, found)
	}
	if r.err != nil {
		return nil, r.err
	}
	return r, nil
}

func (r *reader) fail(err error) {
	if r.err == nil {
		r.err = err
	}
}

func (r *reader) take(n int) []byte {
	if r.err != nil {
		return nil
	}
	if n < 0 || n > len(r.b) {
		r.fail(ErrTruncated)
		return nil
	}
	head := r.b[:n]
	r.b = r.b[n:]
	return head
}

func (r *reader) uint32() uint32 {
	if b := r.take(4); b != nil {
		return binary.LittleEndian.Uint32(b)
	}
	return 0
}

func (r *reader) uint64() uint64 {
	if b := r.take(8); b != nil {
		return binary.LittleEndian.Uint64(b)
	}
	return 0
}

func (r *reader) int() int {
	v := r.uint64()
	if v > math.MaxInt32 {
		r.fail(fmt.Errorf("%w: count too large", ErrCorrupt))
		return 0
	}
	return int(v)
}

func (r *reader) float() float64 { return math.Float64frombits(r.uint64()) }

func (r *reader) bool() bool {
	b := r.take(1)
	if b == nil {
		return false
	}
	switch b[0] {
	case 0:
		return false
	case 1:
		return true
	}
	r.fail(fmt.Errorf("%w: invalid flag", ErrCorrupt))
	return false
}

// len reads a count of items of at least minItemBytes each, rejecting it up
// front when the remaining input cannot hold them (no huge allocations).
func (r *reader) len(minItemBytes int) int {
	n := int(r.uint32())
	if r.err == nil && n > len(r.b)/minItemBytes {
		r.fail(ErrTruncated)
	}
	if r.err != nil {
		return 0
	}
	return n
}

func (r *reader) string() string {
	return string(r.take(r.len(1)))
}

// floats reads exactly n floats (the caller knows n).
func (r *reader) floats(n int) []float64 {
	if n > len(r.b)/8 {
		r.fail(ErrTruncated)
	}
	b := r.take(8 * n)
	if b == nil {
		return nil
	}
	out := make([]float64, n)
	for i := range out {
		out[i] = math.Float64frombits(binary.LittleEndian.Uint64(b[8*i:]))
	}
	return out
}
