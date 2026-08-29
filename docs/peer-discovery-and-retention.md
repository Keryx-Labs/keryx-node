# Peer discovery and retention

Why a node is slow to find peers, drops the ones it has, never reaches `--outpeers`, and receives no
inbound connections — and what this changes.

**Scope: local policy only.** No consensus rule, no wire format, no new message type, no
`PROTOCOL_VERSION` bump. Every change is a constant, a scheduling decision, or a transport (HTTP/2)
setting. A patched node and an unpatched one accept the same blocks and speak the same protocol, so
every benefit is unilateral — it does not require the peer to be patched.

## Why now: the v4 PoM proof

The v4 PoM proof is **~442 KiB per block** (~233 KiB for v2). It broke no explicit limit, but it
halved the margin on every timeout and queue inherited from `rusty-kaspa`, which were sized for
blocks of a few KB. Three of the five mechanisms below are upstream behaviour that only becomes
pathological at 10 BPS with a 442 KiB witness.

## The five mechanisms

**1. `Pong` sits behind 44 MB of block bodies.** `Pong` is enqueued on the same single, unprioritized
`outgoing_route` as block bodies. A peer serving an IBD chunk has `IBD_BATCH_SIZE = 99` bodies ahead
of its pong — ~44 MB of head-of-line delay. Upstream allowed a single pong 120 s, so above ~370 KB/s
it made it and below it did not. Then `Flow::launch` closes the router on **any** flow returning
`Err`, so one late pong killed every other flow with that peer, and `record_ping_timeout` counted it
towards a **24 h IP ban**. The v2→v4 proof doubling halved the bandwidth at which a node starts
banning honest peers.

**2. No keepalive on the serving side.** The client set `tcp_keepalive(10 s)`; the server set
nothing. A NAT dropping its state left a half-open inbound connection that only the ping flow would
notice — up to 240 s later, through the one code path that also bans.

**3. The address store drains itself.** `add_address` inserts with `connection_failed_count = 1`, so
a never-tried address is already one strike in; `mark_connection_failure` **removes** above
`MAX_CONNECTION_FAILED_COUNT = 3`; `connect_timeout` was 1 s, below a single SYN retransmit; `ban()`
also called `remove_by_ip`, so a 24 h ban became permanent amnesia; and `ReceiveAddressesFlow` asked
for addresses **once per connection** then exited. A 300 ms-RTT peer fails three 1 s dials and is
erased. Recovery needs fresh gossip, which needs a new connection, which is exactly what a drained
store cannot make.

**4. The connection loop blocks itself.** The dial loop had no round cap, so it could walk all 4096
addresses in serialized rounds of ≤ 16 dials **before** reaching the DNS seeding at the end of the
iteration — the only recovery path when the store is full of dead addresses. A cold start wasted a
full 30 s tick: the first iteration seeded from DNS and then ended, dialing nothing. And the
dial-exclusion set was built from `peer.net_address()`, which for an **inbound** peer is the remote
*ephemeral* port, so it never matched a store entry and the loop kept re-dialing peers it was already
connected to.

**5. Inbound is invisible and evicted at random.** If no local address is publicly routable and UPnP
does not map, the handshake `Version` advertises no address at all — zero inbound, permanently,
reported only as a single `info!` at boot. Over-limit inbound peers were then evicted with
`choose_multiple`, i.e. uniformly at random: a long-lived peer as likely to go as the newcomer that
caused the overflow.

Raising `--outpeers` reaches none of this. It already defaults to 16 (`--maxinpeers` to 128) and the
target is not being *reached*; raising it only adds dial pressure against a store that is emptying
itself.

## What the code does

### Tolerate a slow pong; ban only a peer that never ponged

`PING_INTERVAL` 120 s → 30 s, an explicit `PONG_TIMEOUT` of 60 s, and `MAX_CONSECUTIVE_PING_TIMEOUTS
= 3`. The load-bearing part is the **outstanding-nonce window**: `SendPingsFlow` keeps a `VecDeque`
of up to three unanswered nonces and matches an incoming pong against the whole window, measuring RTT
from the instant that round's ping was actually sent. Without it, tolerating misses would be unsound
— a late pong arrives with a nonce that no longer matches the latest, and upstream treats a nonce
mismatch as a disconnect. The window is what makes "wait longer" and "keep checking the nonce"
compatible.

| | before | after |
|---|---|---|
| slack for one pong before the peer is dropped | 120 s | ~240 s |
| IBD bandwidth below which an honest peer is dropped | ~370 KB/s | ~190 KB/s |
| a genuinely dead *connection* is detected in | ~240 s | ~50 s (HTTP/2 keepalive) |

The ban gate is narrowed to the signature it was written for: a strike is recorded only when
`router.last_ping_duration() == 0`, i.e. the peer has **never** spoken on this connection — connect
silently, occupy an inbound slot, never answer, reconnect. An honest peer stalled behind its own body
validation is excluded structurally, not by a threshold. Thresholds are widened anyway as defence in
depth (3 → 5 strikes, 600 s → 1800 s window).

### Keepalive on both sides

`connect_timeout` 1 s → 8 s, `communication_timeout` 10 s → 30 s, `tcp_keepalive` 10 s → 30 s (now
also on the server), plus HTTP/2 keepalive at 30 s with a 20 s timeout on both sides. HTTP/2 PING is
a transport frame answered by every hyper/tonic peer, so this needs no protocol change and
interoperates with unpatched nodes.

`connect_timeout` is the one to argue about: 1 s is below a single SYN retransmit, so the old value
did not measure reachability, it measured RTT and wrote off anything more than a few hundred
milliseconds away. Round latency stays bounded because dials are `join_all`-parallel and the round
count is now capped.

### The address store stops deleting

`mark_connection_failure` **saturates** instead of removing. The selection weight already handles
this: a written-off address sits at `64` against a fresh address's `64^3` — **4096× less likely** to
be picked — and one success resets it to 0. `decay_connection_failures()` decrements every non-zero
count when the selection iterator is exhausted while the store is non-empty, so a node that was
offline long enough to saturate its whole store does not treat every address as hopeless forever.
`ban()` no longer removes: the filter moved to selection time, honouring the 24 h expiry.

### The connection loop stops blocking itself

Adaptive cadence replaces the fixed 30 s tick — 2 s while starved (`outbound * 2 < target`), 10 s
below target, 30 s when satisfied. The prerequisite that makes the fast cadence safe is a
**per-address dial cooldown** with exponential backoff (30 s, 1 m, 2 m, 4 m, 8 m); without it a 2 s
loop would just hammer the same unreachable addresses. Rounds are capped at 4 per iteration so DNS
seeding is always reached, seeding that learns something new triggers an immediate re-run instead of
waiting for the next tick, and the dial-exclusion set is built from canonical IPs so inbound peers no
longer consume outbound dial slots.

### Periodic address requests, non-fatal on timeout

`ReceiveAddressesFlow` becomes a loop: a 5 s random initial jitter so N peers are not all asked at
once, then every 60 s while the store holds fewer than 64 addresses and every 600 s otherwise. A
timeout logs at `debug!` and continues to the next cycle rather than killing the connection.

**This is backward compatible without a protocol change**: `SendAddressesFlow` on every deployed node
already answers `RequestAddresses` in an unbounded loop — the one-shot behaviour was on the
*receiving* side only. Load is trivial: ≤ 1000 addresses is ~14 KB per peer per 10 minutes.

### Inbound: say it out loud, and evict deliberately

A `warn!` now names the consequence when no local address is routable ("will receive ZERO inbound
connections") and the fix (`--externalip`). Inbound eviction is ranked worst-first: a peer that has
never answered a pong past a 300 s grace (a silent squatter) goes first, then the newest connections
so a burst of arrivals cannot displace established peers, then by how represented the peer's /16 (or
/64) already is among inbound peers. Permanent requests and ban-exempt seeders are never candidates.

A health line is logged once a minute, to make the four symptoms distinguishable in the field:

```
P2P health: outbound=3/16 (v11=2 v10=1 other=0) inbound=0/128 | addresses=41 cooling_down=7 banned=2 | advertised=NONE (...)
```

A small `addresses` means discovery is the bottleneck; a healthy `addresses` with low `outbound`
means dials are failing; `advertised=NONE` or a stuck `inbound=0` means the node is unreachable from
outside.

## Two correctness traps

**A successful dial is not evidence of a usable peer.** Clearing an address's cooldown on a
*successful* dial produced a regression worse than the problem: a session can die seconds after
connecting — most commonly a peer asking for a block body we only hold the header of, which raises
`BlockNotFound` and tears down the whole router — and with the cooldown cleared that peer carried no
backoff at all, so the loop re-dialed it every 2 s under the starved cadence against 30 s before.
The fix inverts the default: a successful dial takes a **provisional hold**, lifted only by
`forgive_stable_peers` for outbound peers that stayed connected past 60 s. A stable peer accumulates
no history and is immediately re-dialable if it later drops; a flapping peer escalates 30 s → 8 m
even though every dial "succeeds".

**Ban matching must normalize the address form.** `AddressKey` maps IPv4 to its v6-mapped form,
`Entry` stores the address as supplied, and `ban` canonicalizes to v4. Comparing `IpAddress` values
directly would let a ban on `1.2.3.4` miss an entry stored as `::ffff:1.2.3.4` — a silently
ineffective ban, in a filter whose whole purpose is to stop re-dialing a banned peer. Both sides go
through `mapped_v6`.

## Verification

```bash
cargo clippy -p keryx-addressmanager -p keryx-connectionmanager -p keryx-p2p-lib -p keryx-p2p-flows --tests
cargo test -p keryx-addressmanager -p keryx-connectionmanager -p keryx-p2p-lib
```

Clippy clean on all four crates; 27 tests pass, including `echo::tests::test_handshake`, which
exercises the new connect timeout and keepalives through a real handshake. The three `DialCooldown`
tests exist specifically to pin the first correctness trap above.

**No live two-node test has been run.** The remaining verification is a bandwidth-limited pair, one
node in IBD from the other: before these changes the expected observation is `SendPingsFlow flow
error: timeout` and a disconnect; after, the connection should survive the whole IBD.

## Known gaps

Deliberately out of scope, in rough priority order:

1. **Inbound handshakes are serialized in the hub event loop.** `hub.rs` awaits
   `initialize_connection` inline against a 4+4+8 s handshake budget, so one stalling inbound peer
   blocks all inbound acceptance for up to ~12 s. Largest remaining item for the inbound symptom; the
   fix is a spawned task behind a `Semaphore`.
2. **Mainnet has one DNS seeder and one hardcoded IP.** Needs real operator-run hosts, not a code
   change.
3. **`Pong` still shares the bulk send queue.** This removes the *damage* from head-of-line blocking,
   not the blocking. The structural fix is a small priority channel merged into the outgoing stream
   with a biased `select!`. It also invites shrinking `outgoing_network_channel_size()` from 131 328
   — at 442 KiB per message that is ~55 GB of theoretical buffering, which is why backpressure never
   surfaces as `OutgoingRouteCapacityReached` and surfaces as pong delay instead.
4. **A `BlockNotFound` while serving costs a connection.** `route_to_flow` intercepts
   `Payload::Reject` before any routing and always returns `Err`, and the receive loop breaks on
   that. Skipping the hash instead is worse: the requester asks one hash per request and waits with a
   120 s timeout. This needs a protocol-level "not found" response, so that "I don't have that body"
   costs a request rather than a connection. It is the churn this change stops *amplifying* but
   cannot stop.

## Deployment

No lockstep of any kind — no miner change, no consensus change, no protocol version. Roll out one
node at a time and read the `P2P health:` line; the symptoms have different fixes here and it is
worth knowing which one a given node actually had. Rollback is safe: the only persistent state
touched is the address store, where rows now survive that used to be deleted, and an older binary
reads it unchanged.
