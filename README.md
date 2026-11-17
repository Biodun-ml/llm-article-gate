# llm-article-gate

Travel blogs publish thousands of pages, and only some of them describe a trip someone could
actually book. This tool decides which articles are worth turning into a bookable trip, then
pulls out the places and activities with a quoted sentence as evidence for each one. It runs
on a local model by default and includes a small labelled set to compare models.

```
url --> URL gate --> fetch + clean --> classify --> extract --> grounding --> result
          |  regex,      httpx,          LLM, JSON    LLM, JSON   drop items whose
          |  no model    disk cache      + 1 retry    + 1 retry   evidence is not
          v                                  |                    in the article
       skipped                               v
                                    not bookable: stop
```

## Install

Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # core + tests
pip install -e ".[trafilatura]"         # better main-content extraction (optional)
pip install -e ".[anthropic,openai]"    # hosted providers (optional)
pip install -e ".[langfuse]"            # tracing (optional)
```

For the default local model, install [Ollama](https://ollama.com) and pull a model:

```bash
ollama pull qwen2.5:7b
```

On a laptop without a GPU a 7B model can take a minute or more per article, longer on the first
call while the model loads. Requests time out after 600 s by default (`OLLAMA_TIMEOUT` to change
it); a timed-out article is counted as failed and the run continues. `qwen2.5:3b` is a faster
option for a quick check.

Copy `.env.example` to `.env` and fill in only what you use. The CLI reads the environment,
so export the variables or use a tool like `direnv`.

## Usage

```bash
article-gate run https://example-blog.com/portugal/lisbon-in-three-days/
article-gate run data/articles/*.txt --model anthropic:claude-sonnet-5-5
article-gate run urls... --model openai:gpt-4o-mini --json > results.json
```

Model specs are `provider:model`:

| spec                     | needs                                              |
|--------------------------|----------------------------------------------------|
| `ollama:qwen2.5:7b`      | Ollama running locally (default)                   |
| `anthropic:<model>`      | `ANTHROPIC_API_KEY`, `[anthropic]` extra           |
| `openai:<model>`         | `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, `[openai]` extra |
| `fake`                   | nothing; keyword heuristic for offline smoke tests |

Sample output, using the `fake` provider so it runs anywhere:

```
$ article-gate run --model fake data/articles/lisbon-three-days.txt data/articles/jollof-recipe.txt \
    https://travel-notes.example.com/tag/portugal/ https://travel-notes.example.com/about/
BOOK  data/articles/lisbon-three-days.txt  [destination_guide, 1.00] 6 travel keywords
      place     Alfama (other)
      place     LX Factory (other)
      place     Sintra (other)
      place     Rossio (other)

NO    data/articles/jollof-recipe.txt  [other, 0.00] 0 travel keywords

SKIP  https://travel-notes.example.com/tag/portugal/  (url gate: taxonomy)

SKIP  https://travel-notes.example.com/about/  (url gate: site_page)

4 inputs: 2 rejected by URL gate (saved at least 2 LLM calls), 1 bookable, 3 LLM calls made
```

With a real model the classification line shows the model's article type and reason, places
carry a type and country, and any item that failed grounding is listed under `dropped`.

Fetched pages are cached as JSON in `.cache/articles/` (change with `--cache-dir`), so
reruns with a different model do not hit the blog again.

## Evaluation

```bash
article-gate eval data/labels.jsonl --models fake
article-gate eval data/labels.jsonl --models ollama:qwen2.5:7b ollama:llama3.1:8b anthropic:claude-sonnet-5-5
```

`data/labels.jsonl` has 15 hand-labelled articles: 9 destination guides or itineraries and
6 non-candidates (news, a product review, a recipe, a personal essay, a tag listing and an
author page). The articles in `data/articles/` are short synthetic pieces written for this
repo, not copies of real blog posts, so the eval runs offline and can be shared. The URLs
use `example.com` and only matter for the URL gate.

Each line looks like:

```json
{"id": "cartagena", "url": "https://travel-notes.example.com/colombia/cartagena-three-days/",
 "file": "articles/cartagena.txt", "is_bookable": true,
 "places": ["Cartagena", "Walled City", "Getsemaní", "Castillo San Felipe de Barajas", "Rosario Islands"]}
```

What is measured, per model:

- **gate P/R/F1**: end-to-end "is this bookable" decision, URL gate included. An article the
  URL gate rejects counts as predicted not bookable.
- **place P/R/F1**: micro-averaged over all articles, comparing grounded places against the
  labelled list. Names are matched after lowercasing, stripping accents and punctuation, and
  dropping a leading "the", so `Jeronimos monastery` matches `Jerónimos Monastery`.
- **invalid JSON**: share of model replies that failed schema validation (each one triggers
  a retry). **failed** counts articles where the retry also failed.
- **avg s/article** and **tokens in/out** when the provider reports them.

#Each eval run also saves `results/<model>.jsonl` (one line per article: the classification,
the extracted places with their evidence, what grounding dropped, timings) and
`results/summary.md` with the table. Pass `--out other_dir` to keep runs apart.

## Results

Only the offline sanity run is included here. Numbers for real models come from running the
command above on your own hardware; I have not published any.

```
$ article-gate eval data/labels.jsonl --models fake
URL gate rejected 2/15 articles before any model call.

model  gate P  gate R  gate F1  place P  place R  place F1  invalid JSON  failed  LLM calls  avg s/article  tokens in/out
-----  ------  ------  -------  -------  -------  --------  ------------  ------  ---------  -------------  -------------
fake   1.00    0.89    0.94     0.74     0.37     0.50      0%            0       21         0.00           n/a
```

The `fake` provider is a keyword counter and a capitalised-words regex. It exists to check the
harness end to end, not as a baseline worth beating.

## Design decisions

**URL gate before the model.** On a real blog crawl a large share of URLs are tag pages,
category archives, pagination, author bios and site pages. A few regexes reject them for
free, which saves an LLM call per URL and removes the easiest false positives (a tag page
for "Portugal" reads a lot like a Portugal guide). The rules live in
`src/article_gate/url_gate.py` and are deliberately conservative: a missed reject costs one
model call, a wrong reject loses an article.

**Classify before extract.** Classification is a short output and stops most non-candidates
after one call. Extraction is the expensive call and only runs on articles that pass.

**Strict JSON with one retry.** Replies are validated with pydantic. If validation fails, the
model gets its own reply and the validation error back and one more chance. A second failure
is recorded, not raised, so one bad article does not stop a batch.

**Grounding drops unsupported items.** Every place and activity must come with a sentence
quoted from the article. If the quote is not in the article text (after normalising case,
accents, quotes and whitespace), or a place's name is not inside its own quote, the item is
dropped. Small models in particular add well-known places the article never mentions; this
catches that without a second model call, and the dropped items are reported so you can see
how often it happens.

**Local model by default.** Classification and extraction from a single article are within
reach of 7-8B models, and running them locally costs nothing per article, which matters when
the input is a whole blog archive. The eval command exists so you can check whether that
trade-off holds for your data before paying for a hosted model.

**One small provider interface.** A provider is anything with `name` and
`complete(system, prompt) -> Completion`. Ollama, Anthropic and OpenAI-compatible servers
each take about twenty lines. Langfuse tracing wraps whichever provider is in use when the
`LANGFUSE_*` keys are set and the extra is installed; otherwise it is not imported.

## Limitations

- Place matching in the eval is exact after normalisation. `Mỹ Sơn sanctuary` does not match
  a label of `Mỹ Sơn`, so place recall is a lower bound.
- 15 articles is enough to catch regressions and compare models roughly, not to make claims
  about small differences.
- Articles are truncated to 12,000 characters before being sent to the model.
- The URL gate rules are tuned for typical WordPress-style blogs. Sites with other URL
  schemes may need extra rules.
- Main-content extraction uses trafilatura when installed and a BeautifulSoup heuristic
  otherwise. The fallback is weaker on pages with heavy sidebars.
- No deduplication of places across articles, and no geocoding; the output is names, not
  coordinates.

## Tests

```bash
pytest
ruff check .
```

Tests use a scripted `FakeProvider` and make no network calls.

## License

MIT
