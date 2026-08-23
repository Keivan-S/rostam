// SPDX-License-Identifier: Apache-2.0

package vtypes

// OrderKind is the value-type of an order_by field, chosen explicitly on the wire
// (NOT inferred at runtime) so the decode is deterministic and there is no
// mixed-type ambiguity.
//
//   - OrderNumeric  (0, the zero value / default): int or float, read into a
//     float64 key. Datetime shares this float64 path.
//   - OrderDatetime (1): a datetime stored as unix-ms in an int Value; read
//     identically to OrderNumeric (int-ms -> float64).
//   - OrderString   (2): a string/keyword field, read into a string key with a
//     lexicographic (string, id) total order + a v3 cursor.
type OrderKind uint8

const (
	OrderNumeric  OrderKind = 0
	OrderDatetime OrderKind = 1
	OrderString   OrderKind = 2
)

// OrderVal is one typed key value in a MULTI-KEY order tuple: a float64 for
// OrderNumeric/OrderDatetime (Num; Str unused) or a string for OrderString (Str;
// Num unused). Kind selects which field is live so the tuple comparator dispatches
// per key (mixed kinds per key are allowed). OrderVal is comparable (no slices).
type OrderVal struct {
	Num  float64
	Str  string
	Kind OrderKind
}

// OrderBy describes a scroll's ordering: paginate the result set by an arbitrary
// NUMERIC (int/float) or DATETIME payload field instead of the default
// id-ascending order.
//
//   - Key         the (possibly dotted) payload field name to order by.
//   - Desc        descending order on the field value when true; ascending when false.
//   - IsDatetime  the field is a datetime stored as unix-ms in an int Value.
//   - Kind        the value-type of the field (OrderNumeric default / OrderDatetime
//     / OrderString).
//   - StartFrom   an optional initial value bound (Qdrant `start_from`). Only
//     meaningful when HasStart.
//   - HasStart    whether StartFrom is set.
//   - ResumeStr/HasResumeStr  the v3 cursor's resume STRING key (OrderString only).
//   - Tail        the MULTI-KEY extension: the ordered secondary/tertiary sort keys.
//     EMPTY/nil Tail ⇒ the single-key path. A non-empty Tail switches the engine
//     onto the TUPLE-LEXICOGRAPHIC total order + the v4 cursor.
//   - ResumeKeys/HasResumeKeys  the v4 cursor's resume TUPLE (one OrderVal per key,
//     including the primary at index 0). Only meaningful when len(Tail)>0 &&
//     HasResumeKeys.
type OrderBy struct {
	Key           string
	Desc          bool
	IsDatetime    bool
	Kind          OrderKind
	StartFrom     float64
	HasStart      bool
	ResumeStr     string
	HasResumeStr  bool
	Tail          []OrderBy
	ResumeKeys    []OrderVal
	HasResumeKeys bool
}
