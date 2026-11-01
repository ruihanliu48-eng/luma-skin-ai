# LumaSkin AI

A single-file Streamlit prototype for cosmetic skin analysis. A signed-in user uploads
a face photo, the app calls the HKBU vision endpoint, and the result is rendered as two
0-10 KPIs, a short description, and an estimated age. Results are stored in a private
per-user SQLite history and used for a trend dashboard and a prototype product
recommendation.

> Cosmetic assessment only. This is not a medical device or diagnosis. Uploaded photos
> are processed in memory and are never written to the database.

## Features

### Core analysis

- Sidebar password field for the HKBU API key. The key is held only in Streamlit session
  state and is never written to SQLite, source files, logs, or the repository.
- Face-photo upload: JPG, PNG, or WebP, maximum 8 MB.
- Pillow preprocessing: EXIF orientation, alpha flattening, RGB conversion, downscale to
  a maximum side of 1280 px, JPEG re-encode.
- Request to `https://genai.hkbu.edu.hk/api/v1/chat/completions` with model
  `gpt-6-luna`.
- Strict four-field JSON response validation.
- Clear errors for HTTP failures, timeouts, invalid JSON, missing fields, and unusable
  images.

### Three additional features

1. **User access management**
   - Registration and login.
   - Users stored in SQLite.
   - Salted `PBKDF2-HMAC-SHA256` password hashing.
   - `st.session_state` holds the active user ID.
   - Every history query is filtered by the session user ID.

2. **Analysis history and trend dashboard**
   - Each successful analysis stores `user_id`, timestamp, both KPIs, description, and
     estimated age.
   - **History & Trends** shows a saved-scan table.
   - Plotly line chart for Skin Cleanliness and Skin Quality over time.
   - Range filter: 7, 30, 90 days, or all time.

3. **Personalized recommendations and simulated catalog**
   - Rule-based recommendations driven by the lower KPI and selected concerns.
   - SQLite product catalog with name, category, concerns, price, and description.
   - Returns 1-3 products with a short reason.
   - Clearly labelled as a prototype/simulated catalog.

## Files

```text
skincare-ai/
├── app.py                  # Complete application
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml         # Theme, upload limit, headless server
├── .vscode/
│   └── launch.json         # F5 starts Streamlit
└── skin_app.db             # Created automatically at runtime; not committed
```

`skin_app.db` is created on first run. The path is relative to the process working
directory, so run Streamlit from the project root. On Streamlit Community Cloud, the
working directory is the repository root. SQLite WAL mode creates `skin_app.db-wal` and
`skin_app.db-shm` while the app is active. Community Cloud storage is ephemeral; see the
deployment section.

## Stack and requirements

- Python 3.12 recommended
- streamlit
- requests
- pandas
- plotly
- Pillow

`requirements.txt` intentionally contains only these five packages.

## Local setup

### Windows PowerShell

```powershell
cd skincare-ai
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### macOS / Linux

```bash
cd skincare-ai
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501`.

### VS Code

1. Open the `skincare-ai` folder.
2. Select `.venv\Scripts\python.exe` as the interpreter.
3. Press `F5` and choose **Streamlit: LumaSkin AI**.

## API key

Enter the HKBU API key in the sidebar field labeled **HKBU API key**. The input is a
password field.

- No API key is hardcoded.
- No `.env` file is required.
- The key is not persisted to SQLite or any file.
- Signing out clears the key from session state.
- The **Analyze** button stays disabled until both a photo and a key are present.

The endpoint and model are fixed in `app.py`:

```python
API_ENDPOINT = "https://genai.hkbu.edu.hk/api/v1/chat/completions"
API_MODEL = "gpt-6-luna"
```

## LLM request

The request is an OpenAI-compatible multimodal chat completion:

```json
{
  "model": "gpt-6-luna",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "..."},
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,..."
          }
        }
      ]
    }
  ],
  "temperature": 0.1,
  "max_tokens": 500,
  "stream": false
}
```

Required response schema:

```json
{
  "skin_cleanliness_kpi": 7.5,
  "skin_quality_kpi": 6.8,
  "skin_condition_description": "Two or three concise English sentences.",
  "estimated_age": 29
}
```

Validation rules:

| Field | Rule |
| --- | --- |
| `skin_cleanliness_kpi` | Number from 0 to 10 |
| `skin_quality_kpi` | Number from 0 to 10 |
| `skin_condition_description` | Non-empty string, maximum 1000 characters |
| `estimated_age` | Integer from 1 to 120 |

The parser accepts one optional Markdown code fence or a JSON object surrounded by
prose, then requires the key set to match exactly. Extra keys, missing keys, string
scores, booleans, nulls, and out-of-range values are rejected. Invalid output is never
saved and no scores are fabricated.

## Database

SQLite schema created by `init_db()`:

```sql
users (
  id INTEGER PRIMARY KEY,
  username TEXT UNIQUE COLLATE NOCASE,
  password_hash TEXT,
  created_at TEXT
)

analyses (
  id INTEGER PRIMARY KEY,
  user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
  created_at TEXT,
  skin_cleanliness_kpi REAL CHECK (0-10),
  skin_quality_kpi REAL CHECK (0-10),
  skin_condition_description TEXT,
  estimated_age INTEGER CHECK (1-120),
  model TEXT
)

products (
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE,
  category TEXT,
  concerns TEXT,          -- JSON array
  price_cents INTEGER,
  description TEXT
)
```

- `analyses(user_id, created_at DESC)` is indexed.
- Foreign keys are enabled on every connection.
- The product catalog is seeded with idempotent upserts.
- No image BLOB or image path is stored.

## Pages

| Page | Purpose |
| --- | --- |
| **Analyze** | Two-column layout: photo upload and submit on the left; result and observation on the right. |
| **History & Trends** | Summary stats, Plotly trend chart, range filter, and saved-scan table. |
| **Recommendations** | Latest KPI or manual KPI mode, concern selection, 1-3 prototype products. |
| **About** | Data handling, prototype limits, and the non-medical disclaimer. |

Authentication is required before the sidebar navigation appears. The sidebar contains
the nav, API key field, and sign-out button.

## Recommendation rules

The prototype uses the lower of the two KPIs to choose a primary focus:

- Lower **Cleanliness**: prioritize cleansers, exfoliants, and treatments.
- Lower **Quality**: prioritize serums and moisturizers.
- Similar scores: prioritize maintenance products and sunscreen.

Selected concerns add weight to products tagged with those concerns. Only one product per
category is selected, capped at three products. The recommendations are explanatory,
rule-based suggestions, not clinical advice.

## Error behavior

| Condition | Behavior |
| --- | --- |
| No API key | Analyze button disabled; sidebar hint shown. |
| `401` / `403` | No retry; show “API key was rejected.” |
| `429` / `5xx` | One retry after a short delay, then a clear error. |
| Timeout / connection error | Clear error; no database write. |
| Invalid JSON | Clear error; raw output in technical-details expander. |
| Missing/extra/wrong field | Clear validation error; no database write. |
| Image invalid or too small | Reject before the API call. |
| Database save failure | Result is displayed with a “not saved” warning. |

## Streamlit Community Cloud

1. Push the repository to GitHub without `skin_app.db` or `.venv`.
2. Create an app at <https://share.streamlit.io>.
3. Set **Main file path** to `app.py`.
4. Select Python 3.12.
5. Deploy.

No secret is configured in the Cloud dashboard because the API key is entered by each
user in the sidebar. The current code intentionally reads no environment variables or
server-side secrets. If you later add server-side key handling, use the Cloud Secrets UI
and never commit the secret.

Community Cloud uses an ephemeral filesystem. `skin_app.db` resets when the app reboots,
redeploys, or wakes from sleep. For durable Cloud storage, replace the SQLite repository
layer with a hosted database and provide its credentials through Cloud Secrets. A mounted
persistent volume would also require an explicit database-path configuration; the current
relative path is not a durable-storage setup.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `401` | Check the API key in the sidebar. |
| `404` | Verify the endpoint URL in `app.py`. |
| `429` or `5xx` | Wait and retry; one retry is automatic. |
| “Model returned malformed JSON” | Retry. Inspect **Technical details** for the raw output. |
| Image rejected | Use JPG/PNG/WebP, at least 200 x 200 px, under 8 MB. |
| Database reset on Cloud | Expected on Community Cloud’s ephemeral filesystem. |
| `database is locked` | Retry after a moment; connections use a 5-second busy timeout. |

## Limitations

- Cosmetic visual estimation only; no diagnosis or treatment advice.
- Lighting, angle, makeup, filters, and image quality affect the model response.
- The product catalog is fictional and prices are illustrative.
- Authentication is session-based; there is no email recovery or password reset.
- The app does not store or display prior uploaded images.
