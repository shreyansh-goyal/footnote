# Footnote: project notes

Footnote answers questions from my own Obsidian notes and shows where each answer came from.
It is a RAG (retrieval-augmented generation) pipeline: find the relevant pieces of my notes, give them to an LLM, and make the LLM cite them.

## The pipeline

```
Obsidian notes (.md)
   │  loader.py       read files, strip [[wikilinks]]            -> Document
   │  chunker.py      split by heading, then by size              -> Chunk
   │  indexer.py      embed chunks, store in Qdrant               -> vectors + payload
   ▼
Qdrant collection "notes"
   │  retrieve()      embed the question, find the top-k chunks  -> list of hits
   │  answer.py       build a numbered prompt, call the LLM       -> answer + sources
   ▼
Answer with [1] [2] citations and a Sources list
```

Two separate jobs share the vector store:
- **Indexing** (write path): runs when notes change.
- **Asking** (read path): runs whenever I have a question. It should never re-index.

## Modules

### schema.py
`Document` is one note: `title`, `text`, `folder`, `source` (the file name), `source_type`.
Everything downstream works with this one shape, so a new source (the `"medium"` type is already allowed) only needs a new loader.

### loader.py
Reads the `.md` files from the configured folders in the vault and removes the `[[ ]]` wikilink brackets so the text reads as plain prose.
Only the listed folders are indexed, never the whole vault.

### chunker.py
Embedding a whole note is a bad idea: one vector for a long note blurs every topic in it together. So notes are cut into chunks.
1. Split by markdown heading, so each chunk belongs to one topic. Lines inside code fences are never treated as headings.
2. Split any section longer than `MAX_CHARS` (1000) with `RecursiveCharacterTextSplitter`, with `CHUNK_OVERLAP` (150) characters shared between neighbours.

Why overlap: a sentence cut at a chunk boundary would lose its context. Overlap means the end of one chunk is repeated at the start of the next, so an idea that straddles the cut is still whole in one of them.

Each `Chunk` keeps `title`, `folder`, `source` and `heading`, which is what makes citations possible later.

### indexer.py
- **Embedding:** NVIDIA `nemotron-3-embed-1b`, 2048 dimensions, cosine distance.
- **Storage:** one Qdrant point per chunk. The vector is the embedding and the payload is the chunk fields plus `doc_hash`.
- **Point IDs:** `uuid5(source + chunk text)`. The same text always gets the same ID, so writing it again replaces the point instead of duplicating it.

#### Incremental indexing
Embedding costs API calls and time, so unchanged notes are skipped.
1. `note_hash(doc)` is a SHA-256 of the note text plus the settings (model, chunk size, overlap). Changing any setting changes every hash, so everything is re-indexed. Otherwise old and new chunks made with different settings would be mixed.
2. `indexed_hashes(client)` scrolls the collection (payload only, no vectors) and returns `source -> doc_hash` for what is already stored.
3. `plan_changes(current, indexed)` compares the two maps:
   - `to_embed`: sources that are new or whose hash differs.
   - `to_remove`: sources stored in Qdrant that no longer exist as notes.
4. For each affected source, delete its old chunks, then embed and upsert the new ones.

Running it twice with no edits prints `0 notes embedded, N unchanged`.

### answer.py
- `relevant()` drops hits under `MIN_SCORE`, so weak matches never reach the model.
- `format_sources()` numbers the hits once and returns two strings: the block for the prompt, and the footnotes for the user. One numbering, so `[2]` always means the same note in both.
- `build_prompt()` tells the model to use only the numbered sources, cite inline like `[1]`, and say "I don't know" if the sources don't contain the answer.
- `answer()` returns "I don't know" without calling the LLM when nothing relevant was retrieved, otherwise the model's text plus a `Sources:` list.

The Sources list is built from my own hit list, not from what the model writes. The model can only point to a source by number, so it can't invent a source name.

### indexer.py (entry point), ask.py, retriever.py, eval_retrieval.py
- `python -m footnote.indexer` is the write path: it ingests new or changed notes. Run it when notes change.
- `python -m footnote.ask` is the read path: it prompts for a question, retrieves, and prints the cited answer. It never ingests, and exits with a clear message if nothing is indexed.
- `retriever.py` holds `retrieve()`, which returns the top-k chunks as plain dicts.
- `eval_retrieval.py` checks retrieval against a set of questions with known answers (see Evaluation).

## Design decisions worth remembering

| Decision | Why |
| --- | --- |
| Chunk by heading first | Keeps one topic per chunk and gives every chunk a heading to cite |
| Overlap between chunks | Ideas that cross a boundary stay intact in at least one chunk |
| Hash includes settings | Changing the chunker or model forces a full re-index, so the store is never mixed |
| Deterministic point IDs | Re-indexing is idempotent: same text, same ID, no duplicates |
| Hashes stored in the payload | No separate registry to keep in sync, at the cost of scrolling the collection |
| Score threshold before the LLM | A weak match gives "I don't know" instead of a confident wrong answer |
| Citations come from my list | The model can't fabricate a source |

## Known limitations

- **`source` is the file name only.** Two notes with the same name in different folders would collide in the index. Using the vault-relative path would fix it.
- **Delete before embed.** For a changed note, old chunks are deleted before the new ones are embedded. If the embedding call fails in between, that note is missing from search until the next run. Safer order: upsert new chunks, then delete only the stale ones.
- **`indexed_hashes` scans every chunk.** It reads payloads only, so it is cheap at this size. A separate manifest (one record per note) would scale better.
- **`MIN_SCORE` and `k` are guesses.** They need a small test set of questions, each with the note that should answer it, to be tuned properly.
- **No tests yet.** `plan_changes` and `format_sources` are pure functions, so they are easy to cover first.
- **File-level matching over-credits retrieval.** Checking only that the right note was retrieved counts a hit even when a different chunk of that note came back. Evaluation must use chunk-level labels (see Evaluation).
- **Chat models get retired.** NVIDIA returned `410 Gone` for `meta/llama-3.1-70b-instruct`, and the model list still showed it as available. `CHAT_MODEL` should come from an environment variable.
- **The chat model is a reasoning model.** Thinking tokens count against the output limit, so a low `max_completion_tokens` can produce an empty answer. Check `finish_reason` when a reply looks wrong.

## Evaluation

### Golden dataset
A fixed set of test cases that works as an answer key. Each case has:
- a **question**,
- the **expected passage**: the source note plus a short key phrase that must appear in the retrieved chunk,
- optionally a **reference answer**, used later to judge the LLM's final answer.

A human writes or approves it, because it defines what "correct" means. An LLM can draft extra questions from chunks, but they get reviewed.

The dataset stays fixed when `k`, `MIN_SCORE`, the prompt, the chunk size or either model changes. That is what makes before and after scores comparable. It only changes when the notes change or when coverage is extended. Change one setting at a time and re-run the same set.

### Chunk-level labels, not file-level
Matching only the source file counts a hit when a different chunk of the right note is retrieved, so the score says 10/10 while real answers fail. Labelling with a key phrase checks the thing that matters: whether the retrieved text contains what is needed to answer. Key phrases also survive re-chunking, whereas chunk IDs (derived from source and text) change whenever `MAX_CHARS` or `CHUNK_OVERLAP` change.

### Don't tune on the whole dataset
A small warning: don't tune on the whole dataset and then report that number as your quality. If you adjust `MIN_SCORE` until all 10 pass, you've fitted the settings to those 10 questions. With a larger set, teams hold some questions back and check only at the end. For a personal project, 15 or 20 questions with a few you never tuned against is enough.

### Layers
1. **Retrieval:** hit rate at k, then MRR, against the golden set.
2. **Generation:** faithfulness (each claim supported by the cited sources), answer relevance, citation correctness, and correct refusals.

## Implementation plan

Order matters: cheap foundations first, measure before tuning, harden while the index is cheap to rebuild, then containerize.

**A. Foundation**
- `git init`, a `.gitignore` (`.venv`, `data/qdrant`, `.env`), and a pinned `pyproject.toml` or `requirements.txt` (keep the `xxhash` and `grpcio` pins).
- Move the vault path, `CHAT_MODEL`, and Qdrant settings into environment variables, with one `get_client()` function.

**B. Evaluation**
- Write 15 to 20 golden questions (source, key phrase, reference answer), and keep a few aside that are never tuned against.
- Generate a larger synthetic set with a different LLM than the one that answers. Feed it a few handwritten examples as few-shot, and ask for both single-note and multi-note questions.
  - Few-shot examples must not come from the held-out questions, or the held-out set is contaminated.
  - Single-note: give it one chunk. The expected passage is that chunk's key phrase.
  - Multi-note: give it chunks from related notes. It has several expected passages, so score the fraction found in the top k, not a single hit.
  - Verify every generated key phrase appears verbatim in the chunk text, and reject cases that fail. Sample-review the rest by hand.
  - Tell the generator to paraphrase and not copy the chunk's wording, so the questions aren't unrealistically easy.
- Update `eval_retrieval.py` to chunk-level matching, and report hit rate and MRR.
- Tune `k`, `MIN_SCORE`, chunk size and overlap one at a time. Re-run the same set after each change.
- Add generation checks on the same questions.

**C. Hardening**
- Make `source` the vault-relative path so same-named notes can't collide. This changes every ID, so re-index afterwards.
- Re-index order: upsert new chunks first, then delete only the stale chunk IDs.
- Tests for `plan_changes`, `format_sources`, `split_by_heading`, and the eval scoring.
- Reject out-of-range citations, and log retrieval scores and `finish_reason`.

**D. Containerize**
- Qdrant as a Docker container; the client connects by URL.
- A Dockerfile for the app, and a compose file with Qdrant plus the app.
- Ingestion as a job (scheduled or on demand); asking as a read-only service behind a small API.

**E. Operate**
- Run the golden set in CI so a change that drops the score is caught before merge.
- Scheduled ingestion, secrets from the environment, and log real questions, adding the bad ones to the golden set.

## Production plan

Qdrant's local mode (`QdrantClient(path=...)`) locks its folder, so only one process can use the data at a time. That is fine for development but it blocks indexing and asking at the same time.

Target setup:
- **Qdrant** runs as its own Docker container, and the code connects with `QdrantClient(url=...)`. Reads and writes can then happen at the same time.
- **Indexer** runs as a job (scheduled, or triggered when notes change).
- **Ask** runs as a long-lived service (a small API) that only reads.
- Configuration (API keys, Qdrant URL, notes path) comes from environment variables, not constants in the code.
- Because the Qdrant client is built in one place, moving from local mode to the container is a one-line change.

With concurrent access, the delete-before-upsert window above becomes a real issue: a question asked mid-reindex would miss that note. Upsert first, then delete only the stale chunk IDs.
