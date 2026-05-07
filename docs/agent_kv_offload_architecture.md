# Agent KV Offload Architecture

This document describes the current implementation of agent-aware KV admission
and offload in this repository. It focuses on the real code paths: the runner
observes OpenHands events, the sidecar decides admission and offload policy, and
the patched vLLM connector owns KV block safety.

The key invariant is:

> The sidecar chooses *whether* to offload; vLLM chooses *when blocks are safe to free*.

## 1. System Architecture

```mermaid
flowchart TB
    tasks[ABC-Bench tasks]

    subgraph RunnerLayer[Runner layer]
        runner[Instrumented OpenHands runner<br/>event hooks and live state]
        pred[Tool duration predictor]
    end

    subgraph PolicyLayer[Policy and telemetry layer]
        sidecar[Sidecar admission controller<br/>policy loop]
        dash[Dashboard plus JSONL and SSE telemetry]
    end

    subgraph VllmLayer[vLLM control layer]
        api[vLLM agent KV routes<br/>offload, release, restore]
        engine[Patched vLLM engine, core, scheduler]
    end

    subgraph ConnectorLayer[KV safety and transfer layer]
        conn[AgentAwareOffloadingConnector]
        gpu[NPU or GPU KV cache<br/>held request blocks]
        worker[Connector worker<br/>async transfer]
        cpu[CPU offload manager]
    end

    tasks -->|launch queued agents| runner
    runner -->|live agent state and event hooks| sidecar
    pred -->|predicted remaining tool seconds| sidecar
    sidecar -->|admission decisions| runner
    sidecar -->|tick reports| dash
    sidecar -->|POST agent KV control| api
    api -->|forward through vLLM client and core| engine
    engine --> conn
    conn -->|hold, release, restore state| gpu
    conn -->|store metadata| worker
    worker -->|GPU KV blocks| cpu
```

**Ownership split**

| Layer | Responsibility |
| --- | --- |
| Runner | Publishes live agent phase/KV state and calls sidecar hooks on `ActionEvent`, `ObservationEvent`, run end, and errors. Assistant messages are treated as provisional until run end because a tool action may still follow. |
| Sidecar | Computes pressure/headroom, classifies tool calls, queues offload attempts, and admits queued agents. |
| vLLM patch | Exposes control endpoints and forwards them to the scheduler/connector. |
| Agent connector | Holds finished agent requests, snapshots block ids/hashes, starts async store jobs, releases safe holds, and restores via prefix lookup. |
| Dashboard | Visualizes pressure, admissions, offloads, failures, and exact/pending freed-memory accounting. |

## 2. Admission Tick Flow

At every sidecar tick, `DynamicAdmissionController.on_tick()` combines vLLM KV
metrics with live agent state.

Key quantities:

- `C`: free KV capacity in GB.
- `s_t`: current average KV GB among active agents.
- `s_prev`: previous tick's active-agent average.
- `w = C / min(s_t, s_prev)`: effective admission headroom, using whichever
  samples are known and including exact same-tick offload freeing when observed.
- `threshold_percent`: pressure-offload threshold. Percent-only pressure from
  vLLM's cache-used gauge can trigger offload even when free GB is unavailable.
- `w_threshold`: strict admission threshold. Agents admit only when `w` is
  greater than `w_threshold`.
  If pressure is already active and `w` cannot be computed yet, admission stays
  blocked until the sidecar has enough headroom data.

```mermaid
flowchart TD
    tick[Sidecar tick]
    metrics[Poll vLLM metrics<br/>free blocks, total blocks, preemptions]
    live[Read live agents<br/>phase, KV blocks, KV GB, queues]
    compute[Compute C, current average, previous average, headroom]
    pressure{free KV at or below<br/>pressure threshold?}
    heap[Build offload heap<br/>long idle agents only]
    score[Score by KV size<br/>times remaining tool time]
    offload[POST offload endpoint<br/>best candidate]
    exact[Poll exact free-block delta<br/>up to configured timeout]
    recompute[Recompute effective headroom<br/>after offload]
    gate{headroom above<br/>admission threshold?}
    cap{active slots available?}
    admit[Admit work<br/>readmit lane first, then fresh]
    block[Keep queued<br/>emit headroom/pressure/cap reasons]

    tick --> metrics
    tick --> live
    metrics --> compute
    live --> compute
    compute --> pressure
    pressure -- yes --> heap --> score --> offload --> exact --> recompute --> gate
    pressure -- no --> gate
    gate -- yes --> cap
    gate -- no --> block
    cap -- yes --> admit
    cap -- no --> block
```

Important consequences:

- `threshold_percent` is not an admission gate. It only decides when pressure
  offload should run.
- `w_threshold` is the admission gate. The default is `2.0`.
- READMITs use a priority lane ahead of fresh tasks.
- Fresh launches are still capped by `max_fresh_admits_per_tick`.
- If `max_active_agents > 0`, active slots cap both fresh admits and readmits.

## 3. Tool-Call Policy

Tool-call policy starts at `ActionEvent`, not at pressure time. The runner calls
`on_tool_call_start()` immediately when an agent begins a tool call.

```mermaid
flowchart TD
    action[OpenHands ActionEvent<br/>tool call starts]
    close[Runner closes reasoning phase<br/>opens tool call phase]
    snapshot[Runner captures live snapshot<br/>tool metadata + KV state]
    classify[Sidecar classifies tool call]
    predict{Predictor returns<br/>remaining seconds?}
    short{below short-call threshold?}
    releaseShort[Release held KV immediately<br/>short tool-call reason]
    idleShort[Mark short idle<br/>not offloadable yet]
    idleLong[Mark long idle<br/>eligible for pressure heap]
    fallback{elapsed above<br/>fallback long threshold?}
    unavailable[Mark predictor unavailable<br/>not offloadable yet]
    agedShort{still resident and<br/>fallback age reached?}

    action --> close --> snapshot --> classify --> predict
    predict -- yes --> short
    short -- yes --> releaseShort --> idleShort
    short -- no --> idleLong
    idleShort --> agedShort
    agedShort -- yes --> idleLong
    predict -- no --> fallback
    fallback -- yes --> idleLong
    fallback -- no --> unavailable
```

When the tool result arrives:

```mermaid
flowchart TD
    obs[ObservationEvent or rejection]
    record[Record tool duration<br/>clear live tool metadata]
    offloaded{Was agent offloaded?}
    wait[Wait if offloaded<br/>block before next LLM call]
    restore[Sidecar readmits<br/>POST restore endpoint]
    release[Release held KV<br/>tool complete reason<br/>clear idle candidate state]
    reason[Open reasoning phase]

    obs --> record --> offloaded
    offloaded -- yes --> wait --> restore --> reason
    offloaded -- no --> release --> reason
```

For non-offloaded long calls, `release_agent_kv()` also removes the agent from
the idle/offload candidate sets and marks its KV policy as `released` before
the runner opens the next reasoning phase. That prevents a concurrent pressure
tick from offloading a tool call whose pinned blocks were already released.
Cleanup paths call `release_agent_kv()` on final assistant messages, run end,
errors, and cancellation. That prevents stale held KV when the normal
Action/Observation sequence is interrupted.

## 4. Safe KV Hold and Async Offload

The safety fix is "held before free." The connector delays vLLM's normal free
path at `request_finished()` for agent-tagged requests, then releases only after
an explicit release or after async offload completes.

```mermaid
flowchart TD
    finish[vLLM scheduler sees finished agent request]
    snapshot[Connector snapshots block ids and block hashes]
    hold[Connector returns delay free]
    reserved[Request KV stays held and never enters free queue]
    choose[Sidecar chooses agent for pressure offload]
    pending[Connector marks agent pending offload]
    meta[Next scheduler pass builds connector metadata]
    job[Connector creates synthetic store job from held snapshot]
    copy[Worker copies GPU KV blocks to CPU offload manager]
    done[Worker reports synthetic job and real request complete]
    release[Connector drops hold and forwards real request completion]
    free[vLLM frees real request blocks normally]

    finish --> snapshot --> hold --> reserved
    reserved --> choose --> pending --> meta --> job --> copy --> done --> release --> free
```

Release without offload uses the same safety boundary:

```mermaid
flowchart TD
    releaseRequest[Runner or sidecar calls release endpoint]
    connector[Connector receives release request]
    check{Request has pending async store?}
    keep[Keep hold until store completion]
    drop[Drop safe held snapshots]
    ids[Return released request ids]
    freeBlocks[vLLM scheduler frees request blocks]

    releaseRequest --> connector --> check
    check -- yes --> keep
    check -- no --> drop --> ids --> freeBlocks
```

No real implementation path relies on `only_ref_cnt_zero`. The safe invariant is
not "recover blocks after ref count reaches zero"; it is "hold before free."

## 5. Agent State Machine

```mermaid
flowchart TD
    queued([Fresh queue])
    admitted([Admitted])
    reasoning([Reasoning<br/>LLM owns the turn])
    tool([Tool call<br/>tool is running])
    short([Idle short<br/>release held KV])
    long([Idle long<br/>offload candidate])
    offloaded([Offloaded pending tool<br/>KV stored / agent still in tool])
    ready([Offloaded ready<br/>tool result is waiting])
    done([Done<br/>run terminal])

    queued -->|launch| admitted
    admitted -->|LLM request| reasoning

    reasoning -->|ActionEvent| tool
    reasoning -->|final assistant message| done
    reasoning -.->|agent thread error<br/>or external cancel| done

    tool -->|predict short| short
    short -->|tool result, error,<br/>reject, or cancel observation| reasoning
    short -->|fallback age reached<br/>while KV remains resident| long

    tool -->|predict long| long
    long -->|tool result, error,<br/>reject, or cancel observation<br/>+ release| reasoning
    long -->|pressure offload accepted| offloaded

    offloaded -->|tool result, error,<br/>reject, or cancel observation| ready
    ready -->|readmit + restore| admitted

    tool -.->|whole run abort<br/>before observation| done
    done --> terminal([End])

    classDef queue fill:#eef2ff,stroke:#6366f1,color:#111827
    classDef active fill:#ecfeff,stroke:#0891b2,color:#111827
    classDef idle fill:#fef9c3,stroke:#ca8a04,color:#111827
    classDef offload fill:#f5f3ff,stroke:#7c3aed,color:#111827
    classDef terminal fill:#fee2e2,stroke:#dc2626,color:#111827

    class queued,admitted queue
    class reasoning,tool active
    class short,long idle
    class offloaded,ready offload
    class done,terminal terminal
```

Tool-call failures are not terminal by themselves. A failed, rejected, or
cancelled tool call is delivered back to the agent as an observation, then the
agent resumes `reasoning` with that result. `done` is reserved for run-level
termination: final assistant completion, an agent/thread-level error, or an
external cancellation before normal observation handling can continue.

State naming in telemetry:

| State / field | Meaning |
| --- | --- |
| `idle_short` | Tool call predicted short; held KV is released and the agent is excluded from pressure offload until fallback age proves it long while KV is still resident. |
| `idle_long` | Tool call is eligible for pressure offload. |
| `offloaded_pending_tool` | KV offload accepted while the tool call is still running. |
| `offloaded_ready` / `offloaded_waiting` | Tool finished; runner blocks before the next LLM call until readmitted. |
| `readmitted` | Restore notification has been sent; next request can load CPU KV by prefix lookup. |

## 6. Public Interfaces

The vLLM patch adds response-level KV telemetry plus three control endpoints:

| Endpoint | Purpose |
| --- | --- |
| `POST /agent_kv_cache/offload` | Queue held snapshots for async CPU KV offload. |
| `POST /agent_kv_cache/release` | Release held KV without CPU offload. Used for short calls, tool completion, final run cleanup, errors, cancellation, and TTL cleanup. |
| `POST /agent_kv_cache/restore` | Notify readmission. Offloaded KV is loaded by normal OffloadingConnector prefix lookup on the next request. |

Important sidecar fields:

| Field | Meaning |
| --- | --- |
| `w` | Effective admission headroom used for this tick's admission gate. |
| `w_threshold` | Strict admission threshold. Agents admit only when `w > w_threshold`. |
| `w_source` | `current` when `w` is based on the tick's initial free KV, or `after_offload` when exact same-tick offload freeing raised it before admission. |
| `w_before_offload` | Optional diagnostic pre-offload headroom, present only when it differs from effective `w`. |
| `admissions[*].w` | Per-admission copy of the gate value used when the event was emitted. |
| `offloads[*].pending` | The endpoint accepted async connector work, but the free-block delta may not be visible yet. |
| `freed_gb_source` | Source of freed-memory accounting: `vllm_free_blocks_delta`, `offload_endpoint`, `pending_async`, or `unavailable_exact`. |
| `kv_blocks` / `kv_gb` | Latest per-agent KV usage mirrored into `_LIVE_AGENTS` from patched vLLM response usage. Sidecar admission sizing uses these values. |
| `resident_kv_blocks` | Native accelerator KV blocks still owned by active or held agent requests and not in vLLM's free queue, reported by connector control responses when available. |
| `offloadable_kv_blocks` | Complete hash-backed connector blocks that can be copied to CPU. This may be lower than resident blocks when the final native block is only partially filled or when connector block coalescing is active. |
| `held_requests` | Number of held request snapshots for the agent. |
| `known_blocks` | Number of known offloadable KV block hashes from held snapshots. |
| `offload_jobs` | Number of async store jobs already pending for the candidate. |

## 7. Accounting and Failure Modes

```mermaid
flowchart TD
    attempt[Offload attempt]
    accepted{Endpoint accepted?}
    exact{Exact free-block delta<br/>observed before timeout?}
    pending[pending async<br/>accepted but async free not visible yet]
    exactok[exact free-block delta<br/>freed GB is exact]
    fail[offload attempt failed]
    timeout[offload request timeout]
    noheld[no held KV blocks for agent]
    release[release path<br/>short, tool complete, final, or error]
    ttl[TTL cleanup<br/>stale hold release]

    attempt --> accepted
    accepted -- yes --> exact
    exact -- yes --> exactok
    exact -- no --> pending
    accepted -- no --> fail
    fail --> timeout
    fail --> noheld
    release --> ttl
```

Operational interpretation:

| Outcome | Interpretation |
| --- | --- |
| `pending_async` | Offload was accepted. Exact freed memory may appear in a later tick after the async store completes and vLLM frees the real request id. |
| `offload_request_timeout` | The sidecar HTTP request exceeded `offload_timeout_s`. Treat it as an offload attempt failure from the sidecar perspective. |
| `no held KV blocks for agent` | The connector has no held finished request snapshot for that agent. There is nothing safe to offload. |
| `unavailable_exact` | The sidecar could not observe exact freed blocks and the endpoint did not identify the operation as pending async. |
| TTL cleanup | Stale holds are released if runner/sidecar cleanup events are missed. |

The dashboard deliberately does **not** fabricate `freed_gb` from a candidate's
estimated `kv_gb`. `freed_gb` is present only when vLLM reports an exact
free-block delta or an endpoint supplies an explicit exact value.

## 8. Implementation Map

| File | Important code paths |
| --- | --- |
| `src/sidecar.py` | `DynamicAdmissionController`, `on_tick()`, `on_tool_call_start()`, `wait_if_offloaded()`, `release_agent_kv()`. |
| `src/run_abc_bench_instrumented.py` | `ActionEvent`, `ObservationEvent`, final assistant cleanup, run-end cleanup, error cleanup. |
| `src/vllm_patches/agent_offloading_connector.py` | `request_finished()`, held snapshots, pending async jobs, `build_connector_meta()`, `update_connector_output()`, release and restore handling. |
| `src/vllm_patches/apply_patches.py` | vLLM endpoints and forwarding through async LLM, core client, engine core, and scheduler. |

## 9. Mental Model

The system is easiest to reason about as two loops:

1. **Policy loop**: runner events and vLLM metrics feed the sidecar. The sidecar
   classifies tool calls, pressure-offloads the best long idle candidate, and
   admits queued work only when headroom is above `w_threshold`.
2. **Safety loop**: vLLM snapshots and holds finished agent requests before
   free. Held KV is either released directly or copied to CPU by the existing
   async connector path before the real request id is allowed to finish.

Those loops meet at the scheduler-backed usage endpoint and three control
endpoints, but the invariant remains inside vLLM: blocks are never copied after
entering the free queue.
