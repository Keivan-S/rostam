// SPDX-License-Identifier: Apache-2.0
package client

// Collection is a typed, name-scoped handle for vector operations against a
// remote Rostam collection. Constructing one performs no I/O and does not
// require the collection to exist.
type Collection struct {
	c    *Client
	name string
}

// Collection returns a handle bound to the named collection.
func (c *Client) Collection(name string) *Collection { return &Collection{c: c, name: name} }
