// SPDX-License-Identifier: Apache-2.0

package logstore

import (
	"encoding/binary"
	"sync"

	hraft "github.com/hashicorp/raft"
)

// Mem is an in-memory LogStore + StableStore for the NoSync path (durability
// comes from replication, not local disk). The log is a contiguous slice; the
// stable keys are a small map. Nothing is serialized — entries are held decoded.
//
// SAFETY: the log and stable state are VOLATILE. A restart loses them, which is
// safe ONLY if a crashed node rejoins as a FRESH member (catching up from the
// leader's snapshot), never resumes in place. Resuming after losing the
// StableStore (currentTerm / votedFor) could double-vote and break raft safety.
type Mem struct {
	mu   sync.RWMutex
	logs []hraft.Log // contiguous: logs[i].Index == logs[0].Index + i
	kv   map[string][]byte
}

// NewMem returns an empty in-memory store.
func NewMem() *Mem { return &Mem{kv: make(map[string][]byte, 8)} }

func (m *Mem) IsMonotonic() bool { return true }
func (m *Mem) Close() error      { return nil }

func (m *Mem) FirstIndex() (uint64, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if len(m.logs) == 0 {
		return 0, nil
	}
	return m.logs[0].Index, nil
}

func (m *Mem) LastIndex() (uint64, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if len(m.logs) == 0 {
		return 0, nil
	}
	return m.logs[len(m.logs)-1].Index, nil
}

func (m *Mem) GetLog(index uint64, out *hraft.Log) error {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if len(m.logs) == 0 || index < m.logs[0].Index || index > m.logs[len(m.logs)-1].Index {
		return hraft.ErrLogNotFound
	}
	*out = m.logs[index-m.logs[0].Index] // Data/Extensions shared read-only; entries never mutated in place
	return nil
}

func (m *Mem) StoreLog(log *hraft.Log) error { return m.StoreLogs([]*hraft.Log{log}) }

func (m *Mem) StoreLogs(logs []*hraft.Log) error {
	if len(logs) == 0 {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, l := range logs {
		m.appendLocked(cloneLog(l))
	}
	return nil
}

func (m *Mem) appendLocked(c hraft.Log) {
	if len(m.logs) == 0 {
		m.logs = append(m.logs, c)
		return
	}
	last := m.logs[len(m.logs)-1].Index
	switch {
	case c.Index == last+1:
		m.logs = append(m.logs, c)
	case c.Index >= m.logs[0].Index && c.Index <= last:
		off := c.Index - m.logs[0].Index
		m.logs = m.logs[:off+1]
		m.logs[off] = c
	default:
		m.logs = append(m.logs[:0], c) // forward gap only after a monotonic clear
	}
}

func (m *Mem) DeleteRange(minIdx, maxIdx uint64) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.logs) == 0 {
		return nil
	}
	first, last := m.logs[0].Index, m.logs[len(m.logs)-1].Index
	if minIdx < first {
		minIdx = first
	}
	if maxIdx > last {
		maxIdx = last
	}
	if minIdx > maxIdx {
		return nil
	}
	switch {
	case minIdx == first && maxIdx == last:
		m.logs = m.logs[:0]
	case minIdx == first: // front truncation — copy so the head is collectable
		m.logs = append([]hraft.Log(nil), m.logs[maxIdx-first+1:]...)
	case maxIdx == last: // tail truncation — reslice; later appends overwrite
		m.logs = m.logs[:minIdx-first]
	default:
		kept := append([]hraft.Log(nil), m.logs[:minIdx-first]...)
		m.logs = append(kept, m.logs[maxIdx-first+1:]...)
	}
	return nil
}

// --- StableStore (term/vote/metadata; cold path) ---

func (m *Mem) Set(key, val []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.kv[string(key)] = append([]byte(nil), val...)
	return nil
}

func (m *Mem) Get(key []byte) ([]byte, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	v, ok := m.kv[string(key)]
	if !ok {
		return nil, nil
	}
	return append([]byte(nil), v...), nil
}

func (m *Mem) SetUint64(key []byte, val uint64) error {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], val)
	return m.Set(key, b[:])
}

func (m *Mem) GetUint64(key []byte) (uint64, error) {
	v, err := m.Get(key)
	if err != nil || len(v) < 8 {
		return 0, err
	}
	return binary.LittleEndian.Uint64(v), nil
}
