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
- **Entry points are mixed.** `indexer.main()` currently indexes and then answers. These should be two commands (see below).

## Next steps

1. Split the entry points: `index` (write path, run when notes change) and `ask` (read path, never calls `index_all`, fails clearly if nothing is indexed).
2. Build a small retrieval check: questions plus the expected note, and report whether it appears in the top k.
3. Run it for real as services.

## Production plan

Qdrant's local mode (`QdrantClient(path=...)`) locks its folder, so only one process can use the data at a time. That is fine for development but it blocks indexing and asking at the same time.

Target setup:
- **Qdrant** runs as its own Docker container, and the code connects with `QdrantClient(url=...)`. Reads and writes can then happen at the same time.
- **Indexer** runs as a job (scheduled, or triggered when notes change).
- **Ask** runs as a long-lived service (a small API) that only reads.
- Configuration (API keys, Qdrant URL, notes path) comes from environment variables, not constants in the code.
- Because the Qdrant client is built in one place, moving from local mode to the container is a one-line change.

With concurrent access, the delete-before-upsert window above becomes a real issue: a question asked mid-reindex would miss that note. Upsert first, then delete only the stale chunk IDs.
