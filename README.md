<p align="center">
  <img src="banner-dark.png" alt="Mole — a deep research agent in Go, exposed over MCP" width="820">
</p>

<p align="center">
  <em>A deep-research agent with an enforced budget, verified quotes, and a privacy
  boundary for local data.</em>
</p>

Ask a question. mole decomposes it, searches, reads sources, extracts claims,
checks each claim against the text it came from, looks for contradictions between
them, and writes an answer with citations. Every model call is reserved against a
budget before it happens and settled after, so the ceiling you set is the ceiling
it hits.

It runs as a single static binary on your machine, uses your own API keys, and
speaks MCP so a coding agent can drive it — either by handing mole a question and
collecting the answer, or, in **toolkit mode**, by doing the reasoning with its own
model while mole supplies the parts that are not model calls.

<p align="center">
  <img src="demo.svg" alt="mole researching a question: planning, 39 claims, two contradictions found, $0.0149 spent" width="900">
</p>

---

## Why mole

Three things mole does that a chat interface with web search does not.

**The budget is enforced, not estimated.** Every call is reserved before it is
made and settled after, against a ledger with non-negative constraints in the
database schema itself. `--usd 0.50` means the run stops at fifty cents. Measured
overshoot across the test corpus is 0%.

**Every claim carries a quote, checked against the source.** A claim whose quote
does not appear verbatim in the page it was mined from is discarded at extraction,
before it can reach an answer. Claims that survive can be re-read against their
source afterwards, and one that turns out not to be supported is marked as such in
the report rather than quietly dropped.

**Your local data stays local.** Point mole at a CSV or a folder and it will
analyse it without the contents leaving your machine: the model chooses a
hypothesis template and column names, mole renders and runs the SQL, and only
aggregates — counts, means, test results, buckets covering at least five records —
are allowed back. `mole crossings` shows you exactly what left.

---

## Install

**Script** — Linux and macOS, amd64 and arm64:

```sh
curl -fsSL https://raw.githubusercontent.com/lajosdeme/mole/main/install.sh | sh
```

Downloads the release archive for your platform, verifies its SHA-256 against the
checksums published with the release, and installs `mole` and `mole-mcp` into
`~/.local/bin` (or `/usr/local/bin` if that is writable). It uses `sudo` only if
the target directory needs it, and `--dry-run` shows what it would do. If piping a
script into a shell makes you uneasy — reasonable — read it first, or use one of
the paths below.

**Homebrew** — macOS and Linux:

```sh
brew install lajosdeme/mole/mole
```

Fully qualified, and it has to be: an unrelated `mole` (a macOS cleanup tool) is in
homebrew/core, so `brew install mole` will always mean that one. Both install a
binary called `mole`, so only one can be linked at a time.

**Arch Linux** — from the AUR:

```sh
yay -S mole-research-bin      # prebuilt release binaries
yay -S mole-research          # build from source
```

Not `mole`: that name and `mole-bin` on the AUR belong to an SSH tunnelling tool
that has held them since 2020. The package installs `/usr/bin/mole` and declares
the conflict, so pacman will tell you rather than overwrite anything.

**Debian and Ubuntu** — `.deb` from the [releases page](https://github.com/lajosdeme/mole/releases):

```sh
curl -fsSLO https://github.com/lajosdeme/mole/releases/latest/download/mole_amd64.deb
sudo dpkg -i mole_amd64.deb
```

An `.rpm` is published for the same platforms.

**From source** — needs Go 1.25+:

```sh
go install github.com/lajosdeme/mole/cmd/mole@latest
go install github.com/lajosdeme/mole/cmd/mole-mcp@latest
```

Or clone and `make install`, which stamps the version so `mole version` reports the
tag rather than `dev`.

Every path installs the same thing: two static binaries with no runtime
dependencies, built `CGO_ENABLED=0`. The database is SQLite, created on first use
under your XDG data directory.

### Configure

You need a search provider and a model provider. Keys live in
`~/.config/mole/config.json`, mode 0600 — never in environment variables that
leak into process listings, and never in `.mcp.json`.

```bash
mole config set search.provider tavily          # or: brave
mole config set search.tavily-key tvly-...

mole config set llm.provider anthropic          # or: openai-compatible
mole config set llm.api-key sk-...
mole config set llm.model claude-sonnet-5
mole config set llm.cheap-model claude-haiku-4-5

mole doctor                                     # verify everything above
```

Any OpenAI-compatible endpoint works — DeepSeek, Ollama, llama.cpp, vLLM, a proxy:

```bash
mole config set llm.provider openai-compatible
mole config set llm.base-url https://api.deepseek.com/v1
mole config set llm.model deepseek-chat
```

A model served from `localhost` is priced at zero and still counted in tokens, so
`--tokens` bounds a self-hosted run that costs no money at all.

#### Model rates

`--usd` is enforced from a rate table, so mole has to know what a call costs.
It ships rates for the models it can select itself — Anthropic's, DeepSeek's,
`gpt-4o` and `gpt-4o-mini` — and refuses a dollar budget for a model it cannot
price, rather than metering it at zero and letting the ceiling silently not
bind.

Declare anything else, in dollars per **million** tokens, straight off the
vendor's pricing page:

```json
{
  "pricing": {
    "anthropic/claude-sonnet-4.6": { "input": 3.00, "output": 15.00 },
    "some/model-v3": { "input": 1.00, "output": 3.00, "cache_read": 0.50 }
  }
}
```

Cache rates are optional and default to the usual contract — a read at a tenth
of input, writes at 1.25x and 2x. A declared rate overrides a built-in one, so a
price that changes is correctable without waiting for a release.

This is also what makes an aggregator like OpenRouter work: point `base_url` and
`model` at it, declare the rates for the models you use, and `--usd` behaves as
it does anywhere else. `mole doctor` reports whether every configured model has
a rate before you try to run anything.

#### Session limits

```bash
mole config set max-wallclock 20m    # default clock; --timeout overrides per run
```

`max_wallclock` is how long a session may run when `--timeout` says nothing. Not
to be confused with `llm.timeout`, which bounds a single model call.

---

## Usage

### Research a question

```bash
mole research "how much electricity does the bitcoin network use?" --usd 0.50
mole research "..." --tokens 200000            # token budget instead of dollars
mole research "..." --max-sources 8 --max-depth 3
mole research "..." --timeout 20m               # wall-clock ceiling for the session
mole research "..." --max-tool-calls 2000       # backstop; 0 sizes one from --max-depth
mole research "..." --json                      # machine-readable result
```

Budget is required, and the two units are mutually exclusive. Only dollar mode can
price a search call; only token mode can bound a model whose rates mole does not
know.

What stops a run is the clock and the money. `--max-tool-calls` is a backstop for
when neither can — a budget estimate that is wrong, a unit that cannot price a
provider — so it is sized far above what a healthy run needs. The export says
which limit ended the session in `session.stopped_because`: `max_wallclock`,
`max_leads`, `max_tool_calls`, or empty when the planner simply finished.

### Ask a follow-up

```bash
mole ask <session-id> "what did the Cambridge estimate say?"
```

Answers from the claims that session already collected. No new searching, no new
spending beyond the one call to phrase the answer.

### Build a dataset instead of prose

```bash
mole research "largest UK supermarket chains and their revenue" \
  --mode dataset \
  --schema 'company:text!,revenue:number=annual revenue in GBP,employees:number' \
  --usd 0.50

mole dataset <session-id> --format csv > chains.csv
mole dataset <session-id> --format json          # every value every source gave
```

`!` marks the field that identifies a row. Rows are merged across sources by fuzzy
key, so `Aldi` and `Aldi UK` become one row with two sources. CSV holds one value
per cell and says so — it carries a source count and a `contested` column naming
the fields the sources disagree about. JSON carries every disagreeing value with
the sources behind each.

### Analyse local data

```bash
mole connect add sales ./exports/sales.csv       # one file
mole connect add exports ./exports               # or a whole folder
mole research "how does spend differ between regions?" \
  --actors local_compute --usd 0.30

mole crossings <session-id>                      # what left the machine
```

CSV, TSV, JSON and JSONL are supported; Parquet is not. The model never sees a
row and never writes SQL — it picks a template and column names, and mole renders
the statement.

### Serve MCP clients

```bash
mole serve
```

Listens on a unix socket, mode 0600, in a private directory, and refuses
connections from any other user. Point a client at the shim:

```json
{
  "mcpServers": {
    "mole": { "command": "mole-mcp" }
  }
}
```

No credentials in that file — the shim forwards to the daemon, which holds them.

### Use the subscription you already pay for (toolkit mode)

```bash
mole serve --toolkit
```

The arrangement above has mole own the model: it plans, mines and writes with your
API key, and the coding agent driving it is pressing a button. Toolkit mode inverts
that. **The agent's model does the reasoning; mole contributes the deterministic
half** — which is the half worth having, and the half that does not care whose model
is on the other side of it.

If you are inside Claude Code or Qwen Code on a subscription, your model tokens are
already paid for. This is the mode for that.

Fourteen tools, each named `mole.<tool>`, alongside the `research.*` tools — the
flag adds a surface rather than replacing one:

| | |
|---|---|
| session | `session_open`, `session_close` |
| retrieval | `search`, `fetch` — through mole's SSRF guard, robots handling and rate limiter |
| evidence | `verify_quote`, `claim_add`, `claims_list`, `citations` |
| local data | `connect_list`, `aggregate` — the privacy boundary, unchanged |
| graph | `pairs_candidates`, `edge_add` |
| dataset | `rows_add`, `dataset` |

### Inspect a run

```bash
mole sessions                # recent sessions and what they cost
mole trace <session-id>      # per-call cost and timing breakdown
mole stats --fetch           # why fetches failed, across sessions
mole export <session-id>     # the whole session as one JSON document
```

`export` is the machine-readable form: the sub-questions and what became of
each, every claim with the sentence it was mined from, the disagreements, the
coverage and the citation receipt. A sub-question spawned to settle a
contradiction carries `verify_depth` and `root_claim_id`, naming the claim it
came from.

---

## How it works

```
question
   ↓  planner            decompose into sub-questions, replan as evidence arrives
   ↓  executor           one lead at a time per worker, reserved and settled
   ↓  actor              search → fetch → extract → mine claims
   ↓                     every claim quote-checked against its source
   ↓  verifier           pair up related claims, adjudicate, build the claim graph
   ↓                     re-read a sample of claims against their sources
   ↓  output             synthesise from claims that survived, with citations
answer
```

Three actor types feed the same graph. **web** searches and reads pages.
**academic** queries Crossref, OpenAlex, arXiv and PubMed, deduplicates by DOI and
prefers open-access full text. **local_compute** runs deterministic SQL over data
you registered and never lets a row reach the model.

Toolkit mode runs the same machinery with the arrows reversed: the agent decides what
to search, what to read and which claims relate, and mole does the quote checking,
the pair retrieval, the merging and the SQL rendering. Both modes share one copy of
each — the same `AcceptRow` for dataset rows, the same aggregation gate, the same
lexical retriever — so a toolkit graph and an autonomous one are built the same way.

---

## Honest numbers

mole grades its own runs. `mole eval <session-id>` prints a scorecard, and any
metric it cannot compute says so instead of quietly reading zero.

| | |
|---|---|
| budget overshoot | **0%** — no run has exceeded its ceiling |
| claim integrity | **100%** — every stored claim carries a source and a verbatim quote |
| citation accuracy | **100%** — every quote found in the source it cites |
| grounding rate | **80%** — of claims re-read against their source, confirmed |
| contradiction precision | **51%** on one judgement, **70%** on two that agree |
| contradiction recall | **98%** on one judgement, **67%** on two that agree |
| duplicate precision | **64%** on one judgement, **88%** on two that agree |
| merge precision / recall | **1.000 / 1.000** on constructed ground truth |
| staleness separation | **not measurable** on a provider that returns no dates |

Contradiction precision and recall trade against each other, and which way to
take the trade is a flag. Confirming an edge — asking a second time and keeping
only agreements — buys 19 points of precision for 31 of recall, measured on 116
hand-labelled pairs. Autonomous runs confirm by default, because there a false
contradiction also buys a follow-up lead nobody is watching. Output a person
reads wants `--confirm-edges duplicates` instead, which keeps every contradiction
the first judgement found: a missed disagreement is invisible, where a false one
is a pair of quotes on the page. Duplicates are confirmed under both of those,
being the one place a second look costs no recall at all.

Recall is an upper bound. Pairs are formed by a lexical retriever, so two
contradicting claims that share few words are never put to the judge and never
counted as missed — this measures the judge, not the pipeline.

Staleness separation says so rather than reading zero. Telling a stale source
from a disagreeing one needs a publication date on both claims, and a provider
that supplies page text instead of dates leaves nothing to compare: across the
corpus, 369 claims carried none. `mole doctor` reports whether the configured
provider dates its results, and `--always-fetch` recovers dates from the pages
themselves at one fetch per source.

---

## Contributing

Bug reports and issues are welcome. Code contributions go through a CLA — see
[CONTRIBUTING.md](CONTRIBUTING.md), which explains what it is for and what it
cannot do.

Maintainers: the release runbook is [RELEASING.md](RELEASING.md).

The one practice this project asks for that most do not: **falsify your own fix**.
After a change, revert the mechanism and confirm the test fails. A test that passes
with the fix removed proves nothing, and several of this project's own tests have
been caught doing exactly that.

```bash
gofmt -l .      # must print nothing
go build ./...
go test ./...   # must be clean, and no new skips
```

---

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
