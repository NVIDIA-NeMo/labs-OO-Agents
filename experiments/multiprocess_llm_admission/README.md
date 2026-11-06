# Multiprocess LLM admission validation

## Research question

Can the public parent-broker controller enforce one application-wide LLM
concurrency ceiling and one exact call cap across child processes?

## Experiment design

The validator starts a real loopback HTTP server representing an
OpenAI-compatible gateway with capacity four. The concurrency comparison uses
eight spawned Python processes, each submitting 25 simultaneous requests. It
runs the same workload without admission and with one parent-owned
`AdmissionBroker` shared by every child.

A second run submits 32 requests from four processes with `max_calls=19`.
Everything is local; the validator does not contact an external model.
The workload uses Python's `spawn` start method, matching the supported process
startup pattern for a parent that already owns the broker's background thread.

## Key metrics

- Successful requests and gateway overloads.
- Peak gateway concurrency.
- Admission errors and call-cap rejections.
- Broker peak activity, queue state, and admitted-call count.

## Run

```bash
uv run python experiments/multiprocess_llm_admission/validate.py
```

The command exits nonzero if the unprotected path does not reproduce overload,
the protected path exceeds capacity or loses a call, or the call cap is not
exact.

## Results

Local run on macOS with Python's `spawn` start method:

| Run | Requested | Successful | Gateway overloads | Cap rejections | Other errors | Peak | Elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unprotected | 200 | 46 | 154 | 0 | 0 | 4 | 8.795 s |
| Parent broker | 200 | 200 | 0 | 0 | 0 | 4 | 4.513 s |
| Parent broker with call cap | 32 | 19 | 0 | 13 | 0 | 4 | 2.437 s |

The unprotected burst saturated both the simulated gateway and its local HTTP
accept path. The parent-broker run completed every request, held the aggregate
peak at four, and finished with zero active or queued leases.

The call-cap run dispatched exactly 19 of 32 requested provider attempts and
rejected the remaining 13 before dispatch. It had zero gateway overloads and
zero other errors, peaked at four concurrent gateway calls, and completed in
2.437 seconds. All validation criteria passed.

These figures validate the branch implementation under a synthetic local
workload. They are not measurements from an external application or inference
gateway.
