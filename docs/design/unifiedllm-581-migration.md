# UnifiedLLM #581 migration and acceptance ledger

Status: partial implementation; this ledger does not certify completion of
[#581](https://github.com/NVIDIA-NeMo/labs-OO-Agents/issues/581).
Inspected November 25, 2026.

## Design baseline and existing work

The issue references [reviewed revision 21](https://pmgb300ws-0375.ipp9c1.colossus.nvidia.com/artifacts/unifiedllm-one-interface-one-adapter-per-api-family-design-draft-fdd07d9f)
(November 10, revision `revision-3fb37a42-1568-48ac-b1d3-1a2ea8e2ec27`).
The same URL now serves revision 24. Revisions 22–23 consolidate capability
ownership and prohibit inherited media permissions after route changes. Revision
24 moves reasoning/sampling/reply settings from requests to loaded views. Keep
that distinction explicit when comparing implementations with revision 21.

The existing draft replacement stack owns the core implementation:

| Stage | Existing PR | Scope to retain and verify |
| --- | --- | --- |
| 1 | [#574](https://github.com/NVIDIA-NeMo/labs-OO-Agents/pull/574) | Interface, shared shell, LiteLLM compatibility and wire equivalence |
| 2 | [#639](https://github.com/NVIDIA-NeMo/labs-OO-Agents/pull/639) | Typed failures and shared retry policy |
| 2b | [#640](https://github.com/NVIDIA-NeMo/labs-OO-Agents/pull/640) | Per-attempt accounting and tracing |
| 4–5 | [#641](https://github.com/NVIDIA-NeMo/labs-OO-Agents/pull/641) | Native Chat, Responses and Messages adapters |
| 3 | [#642](https://github.com/NVIDIA-NeMo/labs-OO-Agents/pull/642) | Requirements and settings at load |
| 3b | [#643](https://github.com/NVIDIA-NeMo/labs-OO-Agents/pull/643) | Runtime, Atom, CLI and ACP wiring |

The inspected stack tip is `95f67ca35d4d529197a21cce0faa345f61b34b54`.
These are draft implementations, not evidence that every issue criterion passes.
In particular, its `Capabilities.tools=True` compatibility default is explicitly
historical optimism; it cannot be treated as Connect verification. The loader
already accepts `adapter:` while Connect verification snapshots did not include
it. The independent evidence change here closes that invalidation gap without
copying or taking over the stack.

## Staged remaining implementation

1. Preserve the stack's boundaries and offline wire gates. Integrate changes
   through their owning PRs rather than creating a second shell or adapter set.
2. Bind Connect evidence to explicit adapter selection. Adding, changing or
   removing `adapter` or the legacy `direct` selector must invalidate routing,
   media, reasoning and session results. Preserve outcomes and legacy snapshots
   without selectors. Snapshotting a selector alone does not prove its transport
   was used: Connect must eventually build the same adapter as the loader.
3. Add a Connect-to-loader evidence projection. Intersect adapter-format support
   with current route evidence and explicit policy; retain verified, declared,
   unknown and unsupported distinctions. Do not promote model-list metadata,
   an accepted request, or adapter support to a successful capability test.
   Cover individual tool-choice modes, typed output mechanism, multi-turn tools,
   reasoning controls/replay, media and numerical limits independently.
4. Resolve configured routes by canonical model identity plus equivalent endpoint
   and settings. Preserve explicit overrides but validate their compatibility.
   Unknown routes must produce actionable errors before any send. For #111,
   cover raw/qualified GPT-5.6 and equivalent aliases selecting a verified
   Responses route without disabling reasoning or removing tools.
5. Complete named-slot binding and atomic switches. Specify decorator
   client/alias/callable and call-site precedence, user parameters named `llm`,
   inspection and rebinding. Check each slot independently and leave its previous
   binding usable after a rejected switch. Decide explicitly whether proposed
   `method_llms`, `llm_roles` and `Role` APIs are superseded; their discussion's
   closure does not make that API decision. Optional media remains optional;
   note/error rendering must leave stored history unchanged.
6. Complete shared adapter contract regressions and migration documentation.
   Prove instruction-role mapping, foreign tool IDs, portable history and scoped
   native replay, typed output recovery, stable cache boundaries, usage/traces,
   disabled SDK retries and deterministic-error handling. JSON-object fallback
   must put the requested schema after the stable prefix, preserve caller
   messages and validate client-side. Include nested and root output types,
   context-budget effects and route overrides from #499's review. JSON mode
   does not provide provider-enforced schema guarantees.
7. Wire Connect's explicit `adapter:` writes and actual probe transport together;
   preserve legacy overrides and invalidate stale evidence. Only then remove
   provider branches and tighten the boundary ratchet. Resolve remaining design
   decisions about defaults, unknown cost, tracing, old replay scopes and native
   Decisions in the owning implementation/docs, with regression evidence.
8. Run separately authorized live validation against the integrated candidate.
   Keep acceptance, typed correctness, reasoning observed/replayed and actual
   cache hits separate. Offline serializer tests cannot establish these live
   properties.

## Configuration migration in this change

No user registry migration is required. Entries without `adapter`/`direct` retain
existing snapshot shape. An explicit selector is now part of probe configuration:
editing it makes previous results historical, even if `transport` is unchanged.
Removing a selector from a stamped entry also requires new checks. Historical
outcomes are preserved; an edit never stamps new evidence.

This change does not enable a new runtime selector. Use the runtime's supported
configuration for its installed revision. When the adapter stack lands, validate
Connect and application calls through the same selected adapter before describing
the entry as verified.

## Related work reconciliation

[#303](https://github.com/NVIDIA-NeMo/labs-OO-Agents/issues/303),
[#20](https://github.com/NVIDIA-NeMo/labs-OO-Agents/issues/20),
[#111](https://github.com/NVIDIA-NeMo/labs-OO-Agents/issues/111),
[#337](https://github.com/NVIDIA-NeMo/labs-OO-Agents/pull/337) and
[#499](https://github.com/NVIDIA-NeMo/labs-OO-Agents/pull/499) are already closed
in favor of #581. Their closure comments establish consolidation, not completed
acceptance. Reuse #337's transport tests/findings and #499's JSON-mode tests and
review concerns. Record final resolving commits and residual scope before closing
#581; do not close the consolidated items again.

## Verification and live authorization

Offline regression command:

```sh
uv run pytest tests/unifiedllm/connect
```

This change's run: **1,013 passed**. This covers Connect's existing mocked-HTTP
suite plus selector invalidation, including end-to-end routing/save verdicts.
It is not a test run of the draft stack.

Live work follows [nooa-connect](../../skills/nooa-connect/SKILL.md): obtain
explicit alias/endpoint, formats/stages, request/token allowance and file-write
scope; inspect an offline plan, then run only authorized stages and inspect their
reports. Do not run discovery, inference, credential changes or registry writes
merely because credentials exist. Live validation remains pending authorization.
