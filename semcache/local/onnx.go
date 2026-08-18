// SPDX-License-Identifier: Apache-2.0
//go:build localembed

package local

// Confirmed onnxruntime_go API (v1.34.0), read from the module source at
// $(go list -m -f '{{.Dir}}' github.com/yalue/onnxruntime_go):
//
//   Shared library path / environment lifecycle:
//     ort.SetSharedLibraryPath(path string)                       // onnxruntime_go.go:70
//     ort.InitializeEnvironment(opts ...EnvironmentOption) error  // :85
//     ort.DestroyEnvironment() error                              // :122
//
//   Tensors (int64 is part of the TensorData constraint via IntData):
//     ort.NewShape(dims ...int64) Shape                           // :564
//     ort.NewTensor[T TensorData](s Shape, data []T) (*Tensor[T], error) // :762
//     (t *Tensor[T]).GetData() []T                                // :694
//     (t *Tensor[_]).GetShape() Shape  // Shape is []int64        // :715
//     (t *Tensor[_]).Destroy() error                             // :682
//
//   Session with dynamic batch/seq dimensions (shape supplied per Run call):
//     ort.NewDynamicAdvancedSession(onnxFilePath string,
//         inputNames, outputNames []string, options *SessionOptions)
//         (*DynamicAdvancedSession, error)                        // :2829 (options may be nil)
//     (s *DynamicAdvancedSession).Run(inputs, outputs []Value) error // :3082
//         A nil entry in outputs is auto-allocated by ORT and replaced in place
//         with a freshly created Go Value (a *Tensor[float32] for a float output).
//     (s *DynamicAdvancedSession).Destroy() error                // :2847
//
// The binding declares the ONNX Runtime C API itself and dlopen()s the shared
// library named by SetSharedLibraryPath at InitializeEnvironment time, so
// building this file needs neither ORT headers nor libs present; only running a
// session does.

import (
	"errors"
	"fmt"
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// Fixed I/O contract shared with the rest of the localembed path.
var (
	ortInputNames  = []string{"input_ids", "attention_mask", "token_type_ids"}
	ortOutputNames = []string{"last_hidden_state"}
)

var ortInit struct {
	sync.Once
	err error
}

// ResolveORTLibPath returns the path to the ONNX Runtime shared library. It
// honors ROSTAM_ONNXRUNTIME_LIB first, then falls back to conventional install
// locations on Linux (.so) and macOS (.dylib). It returns an error naming the
// override env var and how to obtain ORT if none are found.
//
// lookupEnv is injected (os.LookupEnv in production) so callers/tests can
// control the environment.
func ResolveORTLibPath(lookupEnv func(string) (string, bool)) (string, error) {
	if p, ok := lookupEnv("ROSTAM_ONNXRUNTIME_LIB"); ok && p != "" {
		return p, nil
	}
	for _, cand := range []string{
		"/usr/lib/libonnxruntime.so",
		"/usr/local/lib/libonnxruntime.so",
		"/opt/onnxruntime/lib/libonnxruntime.so",
		"/usr/local/lib/libonnxruntime.dylib",
		"/opt/onnxruntime/lib/libonnxruntime.dylib",
		"/opt/homebrew/lib/libonnxruntime.dylib",
	} {
		if _, err := os.Stat(cand); err == nil {
			return cand, nil
		}
	}
	return "", fmt.Errorf("ONNX Runtime shared library not found; set " +
		"ROSTAM_ONNXRUNTIME_LIB to the full path of libonnxruntime.so " +
		"(or .dylib), or install ONNX Runtime from " +
		"https://github.com/microsoft/onnxruntime/releases")
}

// ensureORT sets the shared-library path and initializes the ORT environment
// exactly once per process. Subsequent calls (with any libPath) return the
// result of the first initialization.
func ensureORT(libPath string) error {
	ortInit.Do(func() {
		ort.SetSharedLibraryPath(libPath)
		ortInit.err = ort.InitializeEnvironment()
	})
	return ortInit.err
}

// ortSession is a reusable ONNX Runtime session over a transformer encoder that
// emits last_hidden_state. Batch and sequence length are dynamic and supplied
// per run call.
type ortSession struct {
	sess *ort.DynamicAdvancedSession
}

// newORTSession ensures the ORT environment is initialized against libPath and
// builds a session for the model at modelPath bound to the fixed input/output
// names.
func newORTSession(modelPath, libPath string) (*ortSession, error) {
	if err := ensureORT(libPath); err != nil {
		return nil, fmt.Errorf("initialize ONNX Runtime: %w", err)
	}
	sess, err := ort.NewDynamicAdvancedSession(modelPath, ortInputNames, ortOutputNames, nil)
	if err != nil {
		return nil, fmt.Errorf("create ONNX session for %q: %w", modelPath, err)
	}
	return &ortSession{sess: sess}, nil
}

// run feeds the three int64 input tensors (each shaped [batch, seqLen]) through
// the session and returns the flattened row-major last_hidden_state together
// with its trailing hidden dimension. tokenType is expected to be all zeros for
// single-segment inputs but is passed through verbatim.
func (s *ortSession) run(inputIDs, attnMask, tokenType []int64, batch, seqLen int) (hidden []float32, hiddenDim int, err error) {
	want := batch * seqLen
	for name, data := range map[string][]int64{
		"input_ids":      inputIDs,
		"attention_mask": attnMask,
		"token_type_ids": tokenType,
	} {
		if len(data) != want {
			return nil, 0, fmt.Errorf("%s has %d elements, want batch*seqLen=%d", name, len(data), want)
		}
	}

	shape := ort.NewShape(int64(batch), int64(seqLen))

	idTensor, err := ort.NewTensor(shape, inputIDs)
	if err != nil {
		return nil, 0, fmt.Errorf("build input_ids tensor: %w", err)
	}
	defer destroyValue(idTensor, &err)

	maskTensor, err := ort.NewTensor(shape, attnMask)
	if err != nil {
		return nil, 0, fmt.Errorf("build attention_mask tensor: %w", err)
	}
	defer destroyValue(maskTensor, &err)

	typeTensor, err := ort.NewTensor(shape, tokenType)
	if err != nil {
		return nil, 0, fmt.Errorf("build token_type_ids tensor: %w", err)
	}
	defer destroyValue(typeTensor, &err)

	inputs := []ort.Value{idTensor, maskTensor, typeTensor}
	outputs := []ort.Value{nil} // nil => ORT auto-allocates last_hidden_state

	if err = s.sess.Run(inputs, outputs); err != nil {
		return nil, 0, fmt.Errorf("run session: %w", err)
	}

	out := outputs[0]
	if out == nil {
		return nil, 0, errors.New("session produced no last_hidden_state output")
	}
	defer destroyValue(out, &err)

	outTensor, ok := out.(*ort.Tensor[float32])
	if !ok {
		return nil, 0, fmt.Errorf("last_hidden_state has unexpected type %T, want *ort.Tensor[float32]", out)
	}

	outShape := outTensor.GetShape() // expect [batch, seqLen, hidden]
	if len(outShape) != 3 {
		return nil, 0, fmt.Errorf("last_hidden_state has rank %d, want 3 ([batch, seqLen, hidden])", len(outShape))
	}
	hiddenDim = int(outShape[2])

	// GetData returns the tensor's backing slice, which destroyValue frees on
	// the deferred cleanup below; copy it out so the caller keeps a valid slice.
	src := outTensor.GetData()
	hidden = make([]float32, len(src))
	copy(hidden, src)

	return hidden, hiddenDim, nil
}

// close releases the underlying session. The ORT environment is process-global
// and torn down separately (see ensureORT / ort.DestroyEnvironment at shutdown).
func (s *ortSession) close() error {
	if s == nil || s.sess == nil {
		return nil
	}
	err := s.sess.Destroy()
	s.sess = nil
	return err
}

// destroyValue frees an ORT value, recording the first cleanup failure into
// *errp only when the caller has not already returned an error, so a genuine
// run error is never masked by a teardown error.
func destroyValue(v ort.Value, errp *error) {
	if v == nil {
		return
	}
	if cerr := v.Destroy(); cerr != nil && *errp == nil {
		*errp = fmt.Errorf("destroy ort value: %w", cerr)
	}
}
