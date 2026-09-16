# LLM admission control

`UnifiedLLM` can place an optional ceiling on asynchronous provider attempts.
Calls above the ceiling wait before provider dispatch instead of producing an
unbounded burst at the inference gateway.

The built-in policy is process local. Multiprocess applications can inject an
admission controller whose state is shared by all workers.

```yaml
models:
  reasoning-model:
    model_name: hosted_vllm/Qwen/Qwen3-32B
    api_base: https://inference.example.com/v1
    concurrency_group: shared-inference
    max_in_flight: 16
    queue_timeout: 30.0
```

The same controls can be passed directly to `CompletionClient`,
`ResponsesClient`, or `get_llm_client()`.

## Semantics

- `max_in_flight` limits simultaneous asynchronous provider attempts. It does
  not limit requests or tokens per minute.
- `concurrency_group` shares one ceiling across clients, model aliases, and the
  Chat Completions and Responses paths.
- Without an explicit group, a configured `api_base` is normalized and hashed
  into an opaque endpoint identity. Once one client configures a budget for an
  endpoint, other clients for that endpoint participate automatically.
- Waiters are admitted in FIFO order. `queue_timeout` raises
  `AdmissionTimeoutError` before provider dispatch and is not retried by the
  provider retry policy.
- Each retry reacquires capacity. Retry backoff does not hold a slot.
- A queued cancellation removes the waiter. After dispatch, capacity remains
  occupied until the provider task exits, even if its caller is cancelled.
- Different `max_in_flight` values for the same group fail at client
  construction instead of depending on construction order.
- Synchronous `call()` is unchanged in this version.

## Multiprocess application scope

An application that owns a group of child processes can run one parent broker
and pass its serializable controller configuration to every child:

```python
import asyncio
import multiprocessing

from nooa.unifiedllm import AdmissionBroker, BrokerAdmissionConfig, get_llm_client


def child(config: BrokerAdmissionConfig) -> None:
    async def run() -> None:
        llm = get_llm_client(
            "reasoning-model",
            admission_controller=config.controller(),
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
attempts, including retries, once the broker confirms a lease; it is not a
successful-response counter.

The broker binds only to a numeric loopback address (`127.0.0.0/8` or `::1`)
and grants FIFO, connection-scoped leases. It does not provide TLS, so both the
broker and its serializable controller configuration reject non-loopback hosts.
A queued timeout or cancellation closes the connection before provider
dispatch. If a child exits while holding a lease, its closed connection returns
the concurrency slot. The application must keep the parent broker alive for the
entire run. Start children with Python's `spawn` or `forkserver` context; do not
start the broker's background thread and then create children with `fork`.

A controller reuses a bounded number of idle connections so a long run does not
consume one ephemeral TCP port per provider attempt. Idle connections expire
automatically; a longer-lived process can call `controller.close()` when it is
finished to release them immediately. Active permits remain valid until their
provider attempt exits, then close instead of returning to a closed controller.

`admission_controller` is a Python runtime object and is intentionally not a
YAML field. Supplying it to `get_llm_client()` overrides any process-local
admission settings inherited from the model registry. Direct client
construction rejects mixing an injected controller with `max_in_flight`,
`concurrency_group`, or `queue_timeout`.

Configured attempts add an `llm.queue` event to the active trace span with the
group, outcome, queued flag, wait duration, queue depth, and limit. Generation
harness metrics aggregate admissions, queued attempts, timeouts, cancellations,
maximum depth, and wait-time statistics.

## Scope boundaries

The default registry is process local. If four child processes each configure a
limit of five, the possible aggregate is twenty. The parent broker coordinates
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

LiteLLM 1.97 includes Router entry points for both Chat Completions and
Responses and has deployment-scoped parallel-request controls. This narrow
NOOA layer preserves direct `UnifiedLLM` paths while defining cross-alias
grouping, queue timeout, cancellation, and trace behavior. A future Router
migration should decide which of these contracts remains in NOOA.
