// SPDX-License-Identifier: Apache-2.0
//go:build windows

package cache

// dirSyncSupported is false because Windows exposes no directory fsync: a
// directory handle opened through os.Open cannot be flushed (FlushFileBuffers
// answers ERROR_ACCESS_DENIED), and NTFS carries the rename through its own
// metadata log rather than leaving it for the caller to force. See syncDir in
// compact.go.
const dirSyncSupported = false
