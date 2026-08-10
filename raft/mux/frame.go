// SPDX-License-Identifier: Apache-2.0

package mux

import (
	"encoding/binary"
	"io"
)

const groupIDLen = 4

func writeGroupID(w io.Writer, id uint32) error {
	var buf [groupIDLen]byte
	binary.BigEndian.PutUint32(buf[:], id)
	_, err := w.Write(buf[:])
	return err
}

func readGroupID(r io.Reader) (uint32, error) {
	var buf [groupIDLen]byte
	if _, err := io.ReadFull(r, buf[:]); err != nil {
		return 0, err
	}
	return binary.BigEndian.Uint32(buf[:]), nil
}
