# Single Agent vs Multi Agent (Subagents)

## What Each Architecture Gets

| Aspect | Single Agent (Multi-Method) | Multi Agent (Subagents) |
|--------|---------------------------|------------------------|
| **Context window** | Shared across all methods. Events accumulate. | Fresh per subagent instance. Each phase starts clean. |
| **Conversation continuity** | Natural. LLM remembers prior method calls. | Lost. Must pass structured summaries explicitly. |
| **State sharing** | `self.x` — any method reads attributes directly. | Explicit passing via constructor/method args. |
| **LLM selection** | One LLM for everything. | Different LLM per subagent (cheap for classification, strong for implementation). |
| **Context isolation** | None. Events from method A are visible in method B. | Full. Each subagent sees only what you give it. |
| **Testing** | One class, harder to test phases independently. | Each subagent independently testable. |
| **Complexity** | One file, one class. | Multiple classes, possibly multiple files. |

## What Subagents Don't Share

Each agent instance is isolated. Even when instantiated inside another agent's `execute_python()` call, subagents do NOT share:
- Context blocks — each agent has its own context
- Events/history — each agent starts with empty history
- Conversation state

Data must be passed explicitly via constructor arguments or shared data objects.

## Data Sharing Between Agents

Since subagents don't inherit context or history, you must share data explicitly:

**Via constructor arguments:**
```python
class ImplementerAgent(Agent, llm=some_llm):
    def __init__(self, plan: Plan, **kwargs):
        super().__init__(**kwargs)
        self.plan = plan
        self.context["plan"] = plan.format()  # Set as context block
```

**Via shared data objects (DABStep pattern):**
```python
@dataclass
class SharedContext:
    brainstorm_result: BrainstormResult
    plan: Plan | None = None
    test_results: list[TestResult] = field(default_factory=list)

# Pass to all subagents
ctx = SharedContext(brainstorm_result=result)
implementer = ImplementerAgent(llm=llm, shared=ctx)
```

**For large shared datasets, cache at module level** — don't reload per call:
```python
_DATA_CACHE: dict[str, DataContext] = {}

def get_data(data_dir: str) -> DataContext:
    if data_dir not in _DATA_CACHE:
        _DATA_CACHE[data_dir] = DataContext.from_dir(data_dir)
    return _DATA_CACHE[data_dir]
```

Loading a 27 MB dataset inside `_run_evaluation()` for each of 450 benchmark tasks is a common footgun.

## Pure Orchestrators Don't Need to Subclass `Agent`

If a class has no `...` generation methods — it only sequences calls to real agents — it doesn't need to be an `Agent` subclass. A plain Python class is cleaner:

```python
# Plain Python orchestrator — no Agent subclassing needed
class Pipeline:
    def __init__(self, llm, data):
        self.llm = llm
        self.data = data

    async def run(self, question: str) -> str:
        discovery = DiscoveryAgent(llm=self.llm, data=self.data)
        solver = SolverAgent(llm=self.llm, data=self.data)

        context = await discovery.run(question)
        result = await solver.compute_answer(question, context)
        return result.answer
```

Use `Agent` subclassing when the class itself has generation methods (`...`) or needs framework features like context blocks and tracing.

## When to Use Which

| Scenario | Better Choice | Why |
|----------|--------------|-----|
| **Conversational agent** (chat, TUI) | Single agent | Conversation continuity, natural flow |
| **Benchmark evaluation** (DABStep, tau-bench) | Subagents | No human in loop, context isolation helps accuracy |
| **Cost-sensitive deployment** | Subagents | Different LLMs per phase |
| **Learning/teaching** | Single agent | One class to read and understand |
| **Complex multi-domain** | Subagents | Clean separation of concerns |
| **Multi-turn with user** | Single agent | User conversation stays in history |
| **Parallel independent tasks** | Subagents | Can run concurrently without shared state |
