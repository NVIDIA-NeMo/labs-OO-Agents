# LLM admission control

`AdmissionControl` is an opt-in wrapper that places a ceiling on asynchronous
provider attempts. Calls above the ceiling wait before provider dispatch
instead of producing an unbounded burst at the inference gateway.

The base client and registry remain unchanged. Wrap a client only where a
particular application use needs admission control:

```python
from nooa.unifiedllm import AdmissionControl, AdmissionControlConfig, get_llm_client

base_llm = get_llm_client("reasoning-model")
llm = AdmissionControl(
    base_llm,
    AdmissionControlConfig(
        max_in_flight=16,
        concurrency_group="shared-inference",
        queue_timeout=30,
    ),
)
```

The built-in controller is process local. Multiprocess applications can wrap
each worker's client with a controller whose state is owned by a parent broker.

## Semantics

- `max_in_flight` limits simultaneous asynchronous provider attempts. It does
  not limit request rate or tokens per minute by itself.
- `concurrency_group` shares one ceiling across clients, model aliases, and the
  Chat Completions and Responses paths.
- Without an explicit group, the wrapped client's configured `api_base` is
  normalized and hashed into an opaque endpoint identity. Only wrapped uses
  participate; an unwrapped use of the same client remains unchanged.
- Waiters are admitted in FIFO order. `queue_timeout` raises
  `AdmissionTimeoutError` before provider dispatch and is not retried by the
  provider retry policy.
- Each retry reacquires capacity. Retry backoff does not hold a slot.
- A queued cancellation removes the waiter. After dispatch, capacity remains
  occupied until the provider task exits, even if its caller is cancelled.
- Different concurrency, pacing, or cooldown settings for the same process-local
  group fail when the wrapper is constructed instead of depending on
  construction order.
- Synchronous `call()` is unchanged in this version.

## Shared request pacing and overload cooldown

Concurrency alone cannot bound the rate of short provider attempts: four slots
with fast responses can still generate many requests each second. Add optional
pacing to the same admission group when the application knows the desired
request rate:

```python
llm = AdmissionControl(
    base_llm,
    AdmissionControlConfig(
        max_in_flight=4,
        concurrency_group="shared-inference",
        requests_per_second=2,
        max_cooldown=60,
        queue_timeout=90,
    ),
)
```

`requests_per_second=2` leaves at least half a second between admission grants,
across all clients sharing this group. It uses minimum spacing, so idle time
does not accumulate tokens for a later burst. Rate eligibility and concurrency
capacity are checked together: waiting for pacing or cooldown does not reserve
a concurrency slot. `queue_timeout` covers the entire admission wait, including
time spent waiting for rate eligibility or cooldown. Every NOOA retry passes
through admission again.

`max_cooldown` opts into sharing overload feedback. When a provider attempt
returns a structured HTTP 429, 503, or 529 error with a valid `Retry-After`
header, its permit reports the delay before releasing the concurrency slot.
Later admission grants for the group pause until the shared deadline. Reports
can extend that deadline; a shorter report never shortens an existing pause.
Existing grants, broker offers awaiting acknowledgement, and active provider
attempts are not revoked. An offer made before the feedback may finish its
handshake and dispatch during the pause; cooldown does not guarantee that all
new HTTP requests stop immediately. The original provider error still follows
the existing retry policy. Reporting also happens when retries are disabled or
a caller has cancelled while its provider task is still running. A feedback
failure does not replace the provider error; the built-in broker bounds its
feedback exchange to one second and closes a connection with an unconfirmed
response instead of reusing it.

The parser accepts positive RFC delta-seconds (unsigned ASCII integers) and
HTTP dates. Fractional, signed, nonfinite, malformed, expired, and nonpositive
values are ignored. No cooldown is inferred from an error message, and success
headers do not update the policy. SDK wrappers may retain the structured
headers in an underlying exception: parsing follows explicit causes or
unsuppressed contexts, with a limit of eight exceptions and cycle protection.
The nearest `Retry-After` field remains authoritative even when malformed; a
non-overload failure cannot inherit an earlier overload's headers.
The configured positive `max_cooldown` caps each reported delay: a value of
60 seconds bounds a single server request for a
longer pause, but repeated reports may extend the overall pause. This cap is an
application tradeoff; resuming before the server's requested deadline can still
produce overload errors.

Both settings default to `None`, preserving concurrency-only admission. Pacing
and cooldown share the group's scope: one process for the built-in policy, or
the processes connected to one parent broker. A broker accepts the same optional
settings:

```python
with AdmissionBroker(
    group="analysis-run-123",
    max_in_flight=4,
    requests_per_second=2,
    max_cooldown=60,
) as broker:
    config = broker.controller_config(queue_timeout=90)
    # Pass config to every worker and wrap each worker's client as below.
```

External controllers retain the `AdmissionController.acquire(observer)` and
`AdmissionPermit.release()` contract. A permit may additionally implement
`CooldownAdmissionPermit`, whose async `cooldown(delay_s)` method records
feedback. A controller with release-only permits continues to work without
shared cooldown. The application owns the controller's policy and coordination;
adding pacing does not require wrapping the same client twice.

## Multiprocess application scope

An application that owns a group of child processes can run one parent broker
and pass its serializable controller configuration to every child:

```python
import asyncio
import multiprocessing

from nooa.unifiedllm import (
    AdmissionBroker,
    AdmissionControl,
    AdmissionControlConfig,
    BrokerAdmissionConfig,
    get_llm_client,
)


def child(config: BrokerAdmissionConfig) -> None:
    async def run() -> None:
        base_llm = get_llm_client("reasoning-model")
        llm = AdmissionControl(
            base_llm,
            AdmissionControlConfig(controller=config.controller()),
        )
        try:
            await llm.acall([{"role": "user", "content": "Analyze this target"}])
        finally:
            await llm.aclose()

    asyncio.run(run())


if __name__ == "__main__":
    context = multiprocessing.get_context("spawn")
    with AdmissionBroker(
        group="analysis-run-123",
        max_in_flight=4,
        max_calls=100,
    ) as broker:
        config = broker.controller_config(queue_timeout=60)
        workers = [context.Process(target=child, args=(config,)) for _ in range(8)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        print(broker.snapshot())
```

Here, `max_in_flight=4` applies across all eight children. `max_calls=100`
allows at most 100 provider attempts for the lifetime of that broker. It counts
attempts, including retries, after a client acknowledges receiving its lease;
it is not a successful-response counter. A disconnect before that
acknowledgement returns the reservation without consuming the call budget.

The broker binds only to a numeric loopback address (`127.0.0.0/8` or `::1`)
and grants FIFO, connection-scoped leases. It does not provide TLS, so both the
broker and its serializable controller configuration reject non-loopback hosts.
The queue deadline covers connection setup, protocol writes, and waiting for a
lease. A timeout or cancellation before admission closes the connection without
provider dispatch. If a child exits while holding a lease, its closed connection
returns the concurrency slot. The application must keep the parent broker alive
for the entire run. Restarting the same broker object creates a fresh bearer
token, so configurations from an earlier broker lifetime fail closed. Start
children with Python's `spawn` or `forkserver` context; do not start the broker's
background thread and then create children with `fork`.

The broker and its workers must use compatible protocol versions. This
feedback-capable protocol rejects older peers rather than silently ignoring
cooldown. Upgrade the owner and workers together and distribute fresh configs.

A controller reuses a bounded number of idle connections so a long run does not
consume one ephemeral TCP port per provider attempt. Idle connections expire
automatically; a longer-lived process can call `controller.close()` when it is
finished to release them immediately. Active permits remain valid until their
provider attempt exits, then close instead of returning to a closed controller.

`AdmissionControlConfig` accepts either a positive process-local
`max_in_flight` with optional pacing and cooldown settings, or an
application-owned `controller`. It rejects missing local limits and rejects
mixing a controller with local settings. Admission is
deliberately not a model-registry or `UnifiedLLM` construction property, so
route overrides cannot accidentally inherit a stale group and the same base
client can be wrapped differently for separate uses.

Configured attempts add an `llm.queue` event to the active trace span with the
group, outcome, queued flag, wait duration, queue depth, and limit. Generation
harness metrics aggregate admissions, queued attempts, call-cap rejections,
timeouts, cancellations, unavailable-controller errors, maximum depth, and
wait-time statistics.

Pacing and cooldown waits contribute to the same queue wait observations. The
measured attempts are NOOA admission grants, not an exact count of HTTP requests
seen by a remote gateway. Provider SDKs may make their own retries inside one
NOOA attempt; disable such retries through supported provider configuration
when comparing wire request counts with admissions. Event-loop scheduling and
transport buffering can also change when an admitted attempt reaches the
gateway.

## Scope boundaries

The built-in wrapper controller is process local. If four child processes each
wrap their clients with a limit of five, the possible aggregate is twenty. The parent broker coordinates
clients that can connect to that broker, normally processes on one host. It is
not a multi-host distributed limiter. The broker serves active and waiting
connections as tasks on one background event loop, so queued bursts do not
create one operating-system thread per attempt. It remains application-scoped
rather than a long-term coordinator for an unbounded, multi-host swarm.

The application owns broker startup, shutdown, run identity, and configuration
distribution because it knows which workers belong to one run. NOOA owns the
provider-attempt hook, lease lifetime, errors, and observations. A host-wide or
distributed hard ceiling still requires a deployed coordinator, shared data
store, or gateway-side enforcement.

Pacing is a configured request-rate policy, not adaptive concurrency or a token
budget. Shared cooldown reacts only to the supported structured overload errors
and their `Retry-After` headers. Neither feature makes the parent broker a
distributed limiter or proves that remote model work stops when a worker
process exits.

LiteLLM 1.97 includes Router entry points for both Chat Completions and
Responses and has deployment-scoped parallel-request controls. This narrow
NOOA layer preserves direct `UnifiedLLM` paths while defining cross-alias
grouping, queue timeout, cancellation, and trace behavior. A future Router
migration should decide which of these contracts remains in NOOA.
