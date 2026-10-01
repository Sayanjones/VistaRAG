# VistaRAG: Tour Guide AI Assistant

Upload a photo of a landmark, then ask questions about it.

The app figures out which landmark you're looking at by comparing your photo to a small set of reference images. Once it knows, it searches a few text files about that landmark and has an OpenAI model write the answer using only what it found there.

This is a rebuild of the project from [computervisioneng's tutorial repo](https://github.com/computervisioneng/rag-web-app-python-chromadb-openai-streamlit) and its [YouTube walkthrough](https://www.youtube.com/watch?v=_y_QS_RfR9A), with a few fixes (see the end).

## How it works

1. **Recognize the photo.** Reference images are turned into vectors with a pretrained ResNet (via `img2vec_pytorch`) and stored in ChromaDB. Your upload gets the same treatment, and the closest reference image decides the landmark.
2. **Find the relevant text.** Each landmark has its own collection of text chunks, embedded with OpenAI's `text-embedding-3-small`. Your question pulls back the 3 closest chunks.
3. **Answer.** Those chunks go into a prompt that tells the model to answer from that context only.

## Setup

You'll need Python 3.10+ and an OpenAI API key.

```bash
git clone <your-repo-url>
cd tour-guide-rag

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then open .env and paste your key
```

The first run downloads the ResNet weights, so expect a short wait.

## Add your data

Make one folder per landmark inside `data/`. Put a few photos and some text in each:

```
data/
  eiffel_tower/
    photo1.jpg
    photo2.jpg
    history.txt
    facts.txt
  big_ben/
    photo1.jpg
    about.txt
```

Two rules:

- **No dashes in folder names.** Use underscores (`big_ben`, not `big-ben`). The category is read back from an ID that uses a dash as the separator, so a dash in the name breaks recognition.
- **More photos help.** Three to five from different angles works noticeably better than one.

Underscores turn into spaces in the UI, so `big_ben` shows up as "Big Ben".

## Example: the Prague demo

The video uses two Prague landmarks, with the text files taken from their Wikipedia articles:

```
data/
  prague_castle/
    photo1.jpg  photo2.jpg  wikipedia.txt
  astronomical_clock/
    photo1.jpg  photo2.jpg  wikipedia.txt
```

With that data, these are the kinds of questions shown in the video:

| Photo | Question |
| --- | --- |
| Prague Castle | When was it built? |
| Prague Castle | Did something interesting ever happen at the castle? |
| Astronomical Clock | What are the statues around the clock? |
| Astronomical Clock | I notice there are drawings inside the lower clock. What are they? |

Your answers will depend on your own text files and chosen model, so they won't match the video word for word. Because the model is told to answer only from the retrieved text, opinion questions ("do you agree it's beautiful?") tend to get weak or oddly worded replies. That's expected.

## Run it

Build the databases (re-run whenever you change anything in `data/`):

```bash
python create_dbs.py
```

Start the app:

```bash
streamlit run main.py
```

Or test from the command line without the UI:

```bash
python query_data.py path/to/photo.jpg "How tall is it?"
```

## Project layout

| File | What it does |
| --- | --- |
| `create_dbs.py` | Reads `data/` and builds the ChromaDB collections |
| `query_data.py` | Image classification, chunk retrieval, answer generation |
| `main.py` | The Streamlit app |
| `utils.py` | Settings and the image embedding class |
| `.env.example` | Template for your API key and model choice |

## Configuration

Set these in `.env`:

- `OPENAI_API_KEY` (required)
- `LLM_MODEL` (optional, defaults to `gpt-4o-mini`). Use any chat model your account has access to.

## Troubleshooting

**`sqlite3` version error from Chroma.** On some Linux systems the built-in SQLite is too old. `pysqlite3-binary` is in `requirements.txt` for Linux and the code switches to it automatically. On macOS and Windows it isn't needed.

**"Collection imgs does not exist".** You haven't run `python create_dbs.py` yet, or you ran it from a different folder. The `db/` folder is created relative to where you run commands.

**Wrong landmark for an unrelated photo.** See the limits below.

## Limits

- It always picks the closest match. Upload a photo of your cat and it will still name one of your landmarks. A distance cutoff would fix this, but the right value depends on your data, so it isn't set here.
- Image matching is only as good as your reference photos.
- Answers come from the text files only. If the answer isn't in them, you'll get a vague reply or "I don't know".

## What's different from the original

- Images are read as RGB on both the indexing and query side. The original read uploads with OpenCV (BGR) while indexing with Chroma's RGB loader, which quietly skewed matches.
- `create_dbs.py` can be re-run safely (upsert instead of create).
- Chunk IDs are tied to filenames so files can't overwrite each other.
- The SQLite workaround is optional, so it doesn't break installs on Windows and macOS.
- OpenCV and a few unused packages were dropped.
- Answers show which files they came from.

## License

MIT. See `LICENSE` (swap in your name).
