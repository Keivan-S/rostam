// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	puddle "github.com/jackc/puddle/v2"
)

type perServerPool struct {
	// 64-bit atomics first for alignment on 32-bit archs.
	newConnsCount        int64
	lifetimeDestroyCount int64
	idleDestroyCount     int64

	addr string
	cfg  *Config
	pool *puddle.Pool[*Conn]

	healthCheckChan chan struct{}
	closeOnce       sync.Once
	closeCh         chan struct{}
}

func newPerServerPool(addr string, cfg *Config) (*perServerPool, error) {
	p := &perServerPool{
		addr:            addr,
		cfg:             cfg,
		healthCheckChan: make(chan struct{}, 1),
		closeCh:         make(chan struct{}),
	}
	puddleCfg := &puddle.Config[*Conn]{
		MaxSize: cfg.MaxConnsPerServer,
		Constructor: func(ctx context.Context) (*Conn, error) {
			atomic.AddInt64(&p.newConnsCount, 1)
			if cfg.BeforeConnect != nil {
				if err := cfg.BeforeConnect(ctx, addr); err != nil {
					return nil, err
				}
			}
			c, err := dial(ctx, addr, cfg)
			if err != nil {
				return nil, err
			}
			if cfg.AfterConnect != nil {
				if hookErr := cfg.AfterConnect(ctx, c); hookErr != nil {
					_ = c.close()
					return nil, hookErr
				}
			}
			return c, nil
		},
		Destructor: func(c *Conn) {
			if cfg.BeforeClose != nil {
				cfg.BeforeClose(c)
			}
			_ = c.close()
		},
	}
	pp, err := puddle.NewPool(puddleCfg)
	if err != nil {
		return nil, fmt.Errorf("client: puddle.NewPool for %s: %w", addr, err)
	}
	p.pool = pp

	// Warm up MinConns + start health check.
	go func() {
		if cfg.MinConnsPerServer > 0 {
			_ = p.createIdle(int(cfg.MinConnsPerServer))
		}
		p.backgroundHealthCheck()
	}()
	return p, nil
}

// acquire pulls a Conn from puddle, optionally pings stale conns, and runs
// BeforeAcquire hook. May loop to discard unhealthy conns.
func (p *perServerPool) acquire(ctx context.Context) (*puddle.Resource[*Conn], error) {
	for {
		res, err := p.pool.Acquire(ctx)
		if err != nil {
			return nil, err
		}
		c := res.Value()
		if res.IdleDuration() > time.Second {
			if perr := c.ping(ctx, p.cfg.CallTimeout); perr != nil {
				res.Destroy()
				continue
			}
		}
		if p.cfg.BeforeAcquire != nil && !p.cfg.BeforeAcquire(ctx, c) {
			res.Destroy()
			continue
		}
		return res, nil
	}
}

// release returns a Conn to puddle, or destroys it if poisoned or AfterRelease vetoes.
func (p *perServerPool) release(res *puddle.Resource[*Conn]) {
	c := res.Value()
	if c.poisoned {
		res.Destroy()
		return
	}
	if p.cfg.AfterRelease != nil && !p.cfg.AfterRelease(c) {
		res.Destroy()
		return
	}
	res.Release()
}

func (p *perServerPool) close() {
	p.closeOnce.Do(func() {
		close(p.closeCh)
		p.pool.Close()
	})
}

func (p *perServerPool) reset() {
	p.pool.Reset()
}

func (p *perServerPool) stat() Stat {
	s := p.pool.Stat()
	return Stat{
		AcquireCount:         s.AcquireCount(),
		TotalConns:           s.TotalResources(),
		IdleConns:            s.IdleResources(),
		NewConnsCount:        atomic.LoadInt64(&p.newConnsCount),
		LifetimeDestroyCount: atomic.LoadInt64(&p.lifetimeDestroyCount),
		IdleDestroyCount:     atomic.LoadInt64(&p.idleDestroyCount),
	}
}

func (p *perServerPool) backgroundHealthCheck() {
	t := time.NewTicker(p.cfg.HealthCheckPeriod)
	defer t.Stop()
	for {
		select {
		case <-p.closeCh:
			return
		case <-p.healthCheckChan:
			p.checkHealth()
		case <-t.C:
			p.checkHealth()
		}
	}
}

func (p *perServerPool) checkHealth() {
	for {
		if err := p.checkMinConns(); err != nil {
			return
		}
		if !p.checkConnsHealth() {
			return
		}
		select {
		case <-p.closeCh:
			return
		case <-time.After(500 * time.Millisecond):
		}
	}
}

func (p *perServerPool) checkConnsHealth() bool {
	var destroyed bool
	total := p.pool.Stat().TotalResources()
	for _, res := range p.pool.AcquireAllIdle() {
		c := res.Value()
		if c.isExpired() && total > p.cfg.MinConnsPerServer {
			atomic.AddInt64(&p.lifetimeDestroyCount, 1)
			res.Destroy()
			destroyed = true
			total--
		} else if res.IdleDuration() > p.cfg.MaxConnIdleTime && total > p.cfg.MinConnsPerServer {
			atomic.AddInt64(&p.idleDestroyCount, 1)
			res.Destroy()
			destroyed = true
			total--
		} else {
			res.ReleaseUnused()
		}
	}
	return destroyed
}

func (p *perServerPool) checkMinConns() error {
	toCreate := p.cfg.MinConnsPerServer - p.pool.Stat().TotalResources()
	if toCreate <= 0 {
		return nil
	}
	return p.createIdle(int(toCreate))
}

func (p *perServerPool) createIdle(n int) error {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		go func() {
			err := p.pool.CreateResource(ctx)
			if err == puddle.ErrNotAvailable {
				err = nil
			}
			errs <- err
		}()
	}
	var first error
	for i := 0; i < n; i++ {
		if e := <-errs; e != nil && first == nil {
			cancel()
			first = e
		}
	}
	return first
}
