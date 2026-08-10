// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"reflect"
	"testing"
)

func TestDecidePBPromotions(t *testing.T) {
	const timeout = 8_000_000_000 // 8s in ns
	const now = 100_000_000_000

	cases := []struct {
		name   string
		shards []pbShardLiveness
		want   []pbPromotion
	}{
		{
			name: "live primary — no promotion",
			shards: []pbShardLiveness{
				{shardID: 0, epoch: 1, primary: "n1", isr: []string{"n1", "n2"}, lastRenewNs: now - 1_000_000_000},
			},
			want: nil,
		},
		{
			name: "silent primary with an ISR survivor — promote lowest-id survivor",
			shards: []pbShardLiveness{
				{shardID: 0, epoch: 3, primary: "n1", isr: []string{"n1", "n3", "n2"}, lastRenewNs: now - 20_000_000_000},
			},
			want: []pbPromotion{{shardID: 0, newEpoch: 4, newPrimary: "n2"}},
		},
		{
			name: "silent primary, ISR is only the dead primary — NO promotion (stay down)",
			shards: []pbShardLiveness{
				{shardID: 0, epoch: 1, primary: "n1", isr: []string{"n1"}, lastRenewNs: now - 20_000_000_000},
			},
			want: nil,
		},
		{
			name: "silent primary, empty ISR — NO promotion",
			shards: []pbShardLiveness{
				{shardID: 0, epoch: 1, primary: "n1", isr: nil, lastRenewNs: now - 20_000_000_000},
			},
			want: nil,
		},
		{
			name: "unseeded shard (no primary) — skipped",
			shards: []pbShardLiveness{
				{shardID: 0, epoch: 0, primary: "", isr: nil, lastRenewNs: 0},
			},
			want: nil,
		},
		{
			name: "never-renewed silent primary (lastRenew 0) with survivor — promoted",
			shards: []pbShardLiveness{
				{shardID: 5, epoch: 2, primary: "n2", isr: []string{"n2", "n1"}, lastRenewNs: 0},
			},
			want: []pbPromotion{{shardID: 5, newEpoch: 3, newPrimary: "n1"}},
		},
		{
			name: "boundary: exactly at timeout is still live (<=)",
			shards: []pbShardLiveness{
				{shardID: 0, epoch: 1, primary: "n1", isr: []string{"n1", "n2"}, lastRenewNs: now - timeout},
			},
			want: nil,
		},
		{
			name: "mixed: one live, one failed",
			shards: []pbShardLiveness{
				{shardID: 0, epoch: 1, primary: "n1", isr: []string{"n1", "n2"}, lastRenewNs: now - 1_000_000_000},
				{shardID: 1, epoch: 1, primary: "n2", isr: []string{"n2", "n3"}, lastRenewNs: now - 20_000_000_000},
			},
			want: []pbPromotion{{shardID: 1, newEpoch: 2, newPrimary: "n3"}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := decidePBPromotions(tc.shards, now, timeout, nil)
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("decidePBPromotions = %+v, want %+v", got, tc.want)
			}
		})
	}
}
