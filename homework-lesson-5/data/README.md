# Дані для RAG

Покладіть сюди файли **PDF**, **TXT** або **MD**, які мають потрапити в локальну базу знань.

Потім з каталогу `homework-lesson-5` виконайте:

```bash
python ingest.py
```

Буде створено папку `index/` (FAISS + `chunks.pkl` для BM25).
