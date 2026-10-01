# JalRakshak – Bhagalpur (Bihar) flood early warning

Free, full-stack, no passwords, no external AI API. One FastAPI service serves the API and the web app.

**Real data used:** satellite flood-inundation records (Bhagalpur, 16 blocks, 22 months 2021-25) train the model; Census 2011 block population drives exposure; Open-Meteo gives live rain (per block) and Ganga discharge.
**Own AI:** random-forest flood model, plus our own text models (report reader, severity, needs, chat intent) in Hindi / English / Hinglish.

## A. Run locally (5 min)
```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # edit values; then load them: export $(grep -v '^#' .env | xargs)   (Windows: set each one)
python scripts/train.py && python scripts/train_nlp.py
python -m pytest tests          # 10 tests
uvicorn app.main:app --reload   # open http://localhost:8000
```
With `DEV_SHOW_OTP=1` and no SMTP, the sign-in code is shown on screen (demo only).

## B. Load the official census (do this next)
1. censusindia.gov.in → Primary Census Abstract → CD Block wise → Bihar → Bhagalpur (file `PCA_CDB_1022_F_Census.xls`).
2. `pip install xlrd` then `python scripts/prep_census.py PCA_CDB_1022_F_Census.xls`
3. `python scripts/train.py`
This adds households, children 0-6, illiteracy, agri-labour (vulnerability index) and **verifies the 13 provisional block names**.

## C. Deploy free (Render + Neon + Brevo + cron-job.org)
1. **GitHub**: create a repo, push this folder.
2. **Neon** (neon.tech, free): new project → copy the connection string (`postgresql://…sslmode=require`).
3. **Email**: Brevo (free 300/day) → SMTP & API → SMTP keys: host `smtp-relay.brevo.com`, port 587, login + key. (Gmail works too: smtp.gmail.com, 587, app password.) Verify your sender address in Brevo.
4. **Render** (render.com, free): New → Blueprint → pick the repo (uses `render.yaml`). Set env vars: `DATABASE_URL` (Neon), `ADMIN_EMAILS` (your email), `SMTP_HOST`, `SMTP_USER`, `SMTP_PASS`, `MAIL_FROM`. Keep `DEV_SHOW_OTP=0` once email works.
5. Open the Render URL → Account → Sign up with your admin email → enter the emailed code.
6. **Automatic alerts**: cron-job.org (free) → new job every 15 min → `https://YOUR-APP.onrender.com/api/cron/tick?key=YOUR_CRON_KEY` (value from Render env `CRON_KEY`). This also keeps the free server awake.
7. Health check: `/api/health`, API docs: `/docs`.

## D. Optional upgrades
- Real roads: `pip install osmnx && python scripts/build_roads.py` (saves `data/roads.pkl`, used automatically).
- Replace `data/shelters.csv` / `data/units.csv` with the DDMA Bhagalpur lists.
- Stored rain/river snapshots (table `snapshots`) are collected automatically for future retraining.

## Honest limitations (say these in your pitch)
- Flood records have no rainfall history → the rain/river adjustment is hand-set, not learned (model card says so). Validation: leave-one-year-out AUC ≈ 0.87.
- Report/chat NLP is trained on templated text; real-world accuracy will be lower.
- Shelters, response units and block centroids are placeholders/approximate; 13 block-ID→name links are provisional until step B.
- Decision support only; does not replace IMD / BSDMA / DDMA warnings. Verify helpline numbers (112, 1077, 1078) locally.
