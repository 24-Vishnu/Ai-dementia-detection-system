import os, json, time, random, hashlib, sqlite3, warnings, tempfile, traceback
from datetime import datetime, timedelta
from functools import wraps
 
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
 
warnings.filterwarnings("ignore")
 
app = Flask(__name__)
app.secret_key = "mindguard_secret_key_2024"
CORS(app, supports_credentials=True, origins=["*"],
     allow_headers=["Content-Type","Authorization"],
     methods=["GET","POST","PUT","DELETE","OPTIONS"])
 
# ──────────────────────────────────────────────────────────────────────────────
# 1.  LOAD ML MODELS
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
 
ml_model   = None
ml_scaler  = None
THRESHOLD  = 0.25          # from config.json
FEATURE_NAMES = []
 
def _load_models():
    global ml_model, ml_scaler, THRESHOLD, FEATURE_NAMES
    try:
        import joblib
        ml_model  = joblib.load(os.path.join(MODEL_DIR, "ensemble_model.pkl"))
        ml_scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        cfg_path  = os.path.join(MODEL_DIR, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            THRESHOLD     = cfg.get("threshold", 0.25)
            FEATURE_NAMES = cfg.get("feature_names", [])
        print(f"✅  Ensemble model   : {type(ml_model).__name__}")
        print(f"✅  Scaler           : {type(ml_scaler).__name__}")
        print(f"✅  Threshold        : {THRESHOLD}")
        print(f"✅  Feature count    : {len(FEATURE_NAMES)}")
    except Exception as e:
        print(f"⚠️   Model load error : {e}")
        print("    Running in HEURISTIC mode (no .pkl loaded).")
 
_load_models()
 
# ──────────────────────────────────────────────────────────────────────────────
# 2.  DATABASE
# ──────────────────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(BASE_DIR, "mindguard.db")
 
def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
 
def _init_db():
    with _db() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name       TEXT    NOT NULL,
            age             INTEGER,
            email           TEXT    UNIQUE NOT NULL,
            phone           TEXT,
            caretaker_name  TEXT,
            caretaker_phone TEXT,
            password_hash   TEXT,
            avatar          TEXT,
            created_at      TEXT    DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS tokens (
            token      TEXT PRIMARY KEY,
            user_id    INTEGER,
            expires_at TEXT
        );
        CREATE TABLE IF NOT EXISTS assessments (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            taken_at        TEXT    DEFAULT (datetime('now')),
            speech_score    REAL,
            cognitive_score REAL,
            overall_risk    REAL,
            risk_level      TEXT,
            dementia_prob   REAL,
            model_used      TEXT    DEFAULT 'ensemble',
            features_json   TEXT,
            report_json     TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """)
 
_init_db()
 
# ──────────────────────────────────────────────────────────────────────────────
# 3.  AUTH HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()
 
def _make_token(user_id: int) -> str:
    tok = hashlib.sha256(f"{user_id}{time.time()}{random.random()}".encode()).hexdigest()
    exp = (datetime.now() + timedelta(days=7)).isoformat()
    with _db() as db:
        db.execute("INSERT OR REPLACE INTO tokens VALUES(?,?,?)", (tok, user_id, exp))
    return tok
 
def _user_from_token(token: str):
    if not token:
        return None
    with _db() as db:
        row = db.execute(
            "SELECT u.* FROM tokens t JOIN users u ON t.user_id=u.id "
            "WHERE t.token=? AND t.expires_at > ?",
            (token, datetime.now().isoformat())
        ).fetchone()
    return dict(row) if row else None
 
def _safe_user(u: dict) -> dict:
    u = dict(u)
    u.pop("password_hash", None)
    return u
 
def auth_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
        user  = _user_from_token(token)
        if not user:
            return jsonify({"error": "Unauthorized"}), 401
        request.current_user = user
        return f(*args, **kwargs)
    return wrapper
 
# ──────────────────────────────────────────────────────────────────────────────
# 4.  FEATURE EXTRACTION  (mirrors ex.py exactly — same 85 features, same order)
# ──────────────────────────────────────────────────────────────────────────────
 
# Exact 85-feature column order from config.json / ex.py
FEATURE_ORDER = [
    # MFCCs (52)
    *[f"mfcc_{i}_{s}" for i in range(1, 14) for s in ["mean","std","max","min"]],
    # Pauses (5)
    "num_pauses","mean_pause_duration","max_pause_duration","total_pause_time","pause_ratio",
    # Speech rate (3)
    "speech_rate","total_duration","num_segments",
    # Pitch (4)
    "pitch_mean","pitch_std","pitch_range","pitch_variation",
    # Energy (4)
    "energy_mean","energy_std","energy_max","energy_min",
    # Spectral (6)
    "spectral_centroid_mean","spectral_centroid_std","spectral_rolloff_mean",
    "spectral_bandwidth_mean","zcr_mean","zcr_std",
    # Prosodic (2)
    "tempo","num_beats",
    # Voice quality (3)
    "harmonic_noise_ratio","harmonic_energy","percussive_energy",
    # Linguistic (6)
    "word_count","utterance_count","ttr","unique_words","avg_words_per_utterance","repetition_ratio",
]  # len == 85
 
 
def extract_acoustic_features(audio_path: str) -> dict:
    """
    Extract the same 85 acoustic + linguistic features used in training (ex.py).
    audio_path: path to a WAV or WebM file saved from the browser.
    Linguistic features are approximated from the transcript text sent alongside.
    """
    try:
        import librosa
    except ImportError:
        raise RuntimeError("librosa not installed.  pip install librosa")
 
    feats = {}
 
    # Load audio — same parameters as ex.py
    y, sr = librosa.load(audio_path, sr=16000, duration=60.0)
 
    # ── 1. MFCCs ─────────────────────────────────────────────────────────────
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i]))
        feats[f"mfcc_{i+1}_std"]  = float(np.std(mfccs[i]))
        feats[f"mfcc_{i+1}_max"]  = float(np.max(mfccs[i]))
        feats[f"mfcc_{i+1}_min"]  = float(np.min(mfccs[i]))
 
    # ── 2. Pauses ─────────────────────────────────────────────────────────────
    frame_len = int(0.025 * sr)
    hop       = int(0.010 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    thresh  = np.percentile(rms, 20)
    silent  = rms < thresh
    pauses  = []
    cur     = 0.0
    for s in silent:
        if s:
            cur += hop / sr
        else:
            if cur > 0.1:
                pauses.append(cur)
            cur = 0.0
    feats["num_pauses"]          = float(len(pauses))
    feats["mean_pause_duration"] = float(np.mean(pauses)) if pauses else 0.0
    feats["max_pause_duration"]  = float(np.max(pauses))  if pauses else 0.0
    feats["total_pause_time"]    = float(np.sum(pauses))  if pauses else 0.0
    feats["pause_ratio"]         = float(np.sum(silent) / max(len(silent), 1))
 
    # ── 3. Speech rate ────────────────────────────────────────────────────────
    onsets   = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    duration = len(y) / sr
    feats["speech_rate"]    = float(len(onsets) / duration) if duration > 0 else 0.0
    feats["total_duration"] = float(duration)
    feats["num_segments"]   = float(len(onsets))
 
    # ── 4. Pitch ─────────────────────────────────────────────────────────────
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    pv = []
    for t in range(pitches.shape[1]):
        idx = mags[:, t].argmax()
        p   = pitches[idx, t]
        if p > 0:
            pv.append(float(p))
    if pv:
        feats["pitch_mean"]      = float(np.mean(pv))
        feats["pitch_std"]       = float(np.std(pv))
        feats["pitch_range"]     = float(np.max(pv) - np.min(pv))
        feats["pitch_variation"] = float(np.std(pv) / np.mean(pv)) if np.mean(pv) != 0 else 0.0
    else:
        feats["pitch_mean"] = feats["pitch_std"] = feats["pitch_range"] = feats["pitch_variation"] = 0.0
 
    # ── 5. Energy ─────────────────────────────────────────────────────────────
    feats["energy_mean"] = float(np.mean(rms))
    feats["energy_std"]  = float(np.std(rms))
    feats["energy_max"]  = float(np.max(rms))
    feats["energy_min"]  = float(np.min(rms))
 
    # ── 6. Spectral ──────────────────────────────────────────────────────────
    sc   = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sro  = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    sb   = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    zcr  = librosa.feature.zero_crossing_rate(y)[0]
    feats["spectral_centroid_mean"]  = float(np.mean(sc))
    feats["spectral_centroid_std"]   = float(np.std(sc))
    feats["spectral_rolloff_mean"]   = float(np.mean(sro))
    feats["spectral_bandwidth_mean"] = float(np.mean(sb))
    feats["zcr_mean"]                = float(np.mean(zcr))
    feats["zcr_std"]                 = float(np.std(zcr))
 
    # ── 7. Prosodic ──────────────────────────────────────────────────────────
    tempo_arr, beats = librosa.beat.beat_track(y=y, sr=sr)
    feats["tempo"]     = float(np.squeeze(tempo_arr))
    feats["num_beats"] = float(len(beats))
 
    # ── 8. Voice quality ─────────────────────────────────────────────────────
    harmonic    = librosa.effects.harmonic(y)
    percussive  = librosa.effects.percussive(y)
    h_energy    = float(np.sum(harmonic ** 2))
    p_energy    = float(np.sum(percussive ** 2))
    feats["harmonic_energy"]       = h_energy
    feats["percussive_energy"]     = p_energy
    feats["harmonic_noise_ratio"]  = h_energy / (p_energy + 1e-10)
 
    return feats
 
 
def estimate_linguistic_features(transcript: str) -> dict:
    """
    Estimate linguistic features from the Web Speech API transcript text.
    Used when no CHAT (.cha) file is available (browser recording context).
    Mirrors ex.py's linguistic block as closely as possible.
    """
    words  = [w.lower().strip(".,!?;:") for w in transcript.split() if w.strip()]
    # Naive utterance split on sentence-ending punctuation or long pauses (...).
    utts   = [u.strip() for u in transcript.replace("...", ".").replace("!", ".").replace("?", ".").split(".") if u.strip()]
 
    wc  = len(words)
    uc  = max(len(utts), 1)
    uniq = len(set(words)) if wc > 0 else 0
    ttr  = uniq / wc if wc > 0 else 0.0
    awpu = wc / uc
 
    # Repetition ratio
    if wc > 0:
        from collections import Counter
        counts   = Counter(words)
        repeated = sum(1 for c in counts.values() if c > 1)
        rep_ratio = repeated / max(len(counts), 1)
    else:
        rep_ratio = 0.0
 
    return {
        "word_count":              float(wc),
        "utterance_count":         float(uc),
        "ttr":                     float(ttr),
        "unique_words":            float(uniq),
        "avg_words_per_utterance": float(awpu),
        "repetition_ratio":        float(rep_ratio),
    }
 
 
def build_feature_vector(acoustic: dict, linguistic: dict) -> np.ndarray:
    """
    Merge acoustic + linguistic into the exact 85-dim vector the scaler expects.
    Unknown features default to 0.
    """
    merged = {**acoustic, **linguistic}
    vec    = np.array([merged.get(f, 0.0) for f in FEATURE_ORDER], dtype=np.float64)
    vec    = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 5.  ML PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
 
def _predict_with_model(feature_vec: np.ndarray) -> dict:
    """
    Run the ensemble model and return probabilities + prediction.
    Falls back to heuristic if model not loaded.
    """
    if ml_model is None or ml_scaler is None:
        return _heuristic_predict(feature_vec)
 
    import pandas as pd
    # Wrap in DataFrame with correct column names — avoids sklearn warning
    X_df     = pd.DataFrame([feature_vec], columns=FEATURE_ORDER)
    X_scaled = ml_scaler.transform(X_df)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
 
    probs    = ml_model.predict_proba(X_scaled)[0]   # [P(control), P(dementia)]
    p_dem    = float(probs[1])
    p_ctrl   = float(probs[0])
    predicted = int(p_dem >= THRESHOLD)
 
    return {
        "dementia_probability": p_dem,
        "control_probability":  p_ctrl,
        "prediction":           predicted,
        "threshold_used":       THRESHOLD,
        "model_used":           "ensemble_pkl",
    }
 
 
def _heuristic_predict(feature_vec: np.ndarray) -> dict:
    """
    Fallback heuristic when .pkl files aren't loaded.
    Uses clinically-informed acoustic indicators.
    """
    feat = dict(zip(FEATURE_ORDER, feature_vec))
    score = 0.0
 
    # Pause-related (strongest dementia indicator)
    if feat.get("num_pauses", 0) > 10:    score += 0.25
    if feat.get("pause_ratio", 0) > 0.25: score += 0.15
    if feat.get("mean_pause_duration", 0) > 0.3: score += 0.10
 
    # Speech rate (slow = concern)
    if feat.get("speech_rate", 5) < 2:    score += 0.15
 
    # Linguistic (low diversity = concern)
    if feat.get("ttr", 1) < 0.3:          score += 0.10
    if feat.get("repetition_ratio", 0) > 0.3: score += 0.10
 
    # Word count very low
    if feat.get("word_count", 50) < 20:   score += 0.10
 
    p_dem  = min(score, 0.95)
    p_ctrl = 1.0 - p_dem
 
    return {
        "dementia_probability": p_dem,
        "control_probability":  p_ctrl,
        "prediction":           int(p_dem >= THRESHOLD),
        "threshold_used":       THRESHOLD,
        "model_used":           "heuristic",
    }
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 6.  SCORING HELPERS  (map ML output → frontend TestResult fields)
# ──────────────────────────────────────────────────────────────────────────────
 
def _risk_level(overall_risk: float) -> str:
    if overall_risk <= 40:  return "Low"
    if overall_risk <= 70:  return "Moderate"
    return "High"
 
 
def _compute_scores(pred: dict, cog_answers: dict, total_words: int = 50) -> dict:
    """
    Combine ML probability with cognitive-test answers to produce
    speechScore / cognitiveScore / overallRisk in the same format
    as the original frontend calculateScores().
    total_words: actual word count from all 3 speech transcripts combined.
    """
    p_dem = pred["dementia_probability"]

    # ── Override speech risk based on actual word count ───────────────────────
    # The ML model on a zero vector gives ~0.44 — not meaningful for empty input.
    # Use word count as the primary speech risk signal when no audio is provided.
    # Only override when word count is very low — otherwise trust the ML model.
    if total_words == 0:
        speech_risk_override = 95.0   # no speech at all = very high risk
    elif total_words < 10:
        speech_risk_override = 75.0   # very few words
    elif total_words < 30:
        speech_risk_override = None   # partial speech — trust ML
    else:
        speech_risk_override = None   # good speech — fully trust ML model
 
    # ── Cognitive score from the Q&A answers ─────────────────────────────────
    cog_correct = 0
    cog_total   = 0
 
    # Day of week
    cog_total += 1
    days = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
    answered_day = cog_answers.get("dayOfWeek","")
    if answered_day == days[datetime.now().weekday()]: cog_correct += 1.0
    elif answered_day: cog_correct += 0.3

    # Year
    cog_total += 1
    answered_year = str(cog_answers.get("currentYear",""))
    if answered_year == str(datetime.now().year): cog_correct += 1.0
    elif answered_year: cog_correct += 0.3

    # Location
    cog_total += 1
    if str(cog_answers.get("location","")).strip():
        cog_correct += 1.0

    # Math
    cog_total += 2
    if str(cog_answers.get("mathSpent","")).strip() == "700":     cog_correct += 1.0
    elif str(cog_answers.get("mathSpent","")).strip():             cog_correct += 0.2
    if str(cog_answers.get("mathRemaining","")).strip() == "300": cog_correct += 1.0
    elif str(cog_answers.get("mathRemaining","")).strip():         cog_correct += 0.2

    # Animal fluency
    cog_total += 1
    animals = cog_answers.get("animals", [])
    unique_animals = len(set(str(a).lower().strip() for a in animals if a))
    if unique_animals >= 15:   cog_correct += 1.0
    elif unique_animals >= 10: cog_correct += 0.85
    elif unique_animals >= 7:  cog_correct += 0.65
    elif unique_animals >= 4:  cog_correct += 0.45
    elif unique_animals >= 1:  cog_correct += 0.25

    # Object recall
    cog_total += 1
    targets = ["apple","pen","tie","house","car"]
    recalled = sum(
        1 for r in cog_answers.get("objectRecall", [])
        if any(t in str(r).lower() for t in targets)
    )
    cog_correct += recalled / 5.0

    # Clock
    cog_total += 1
    h_ang = cog_answers.get("clockHourAngle", 0)
    m_ang = cog_answers.get("clockMinuteAngle", 0)
    h_target, m_target = 330, 300
    hd = abs(((h_ang % 360) + 360) % 360 - h_target)
    md = abs(((m_ang % 360) + 360) % 360 - m_target)
    if hd < 30 and md < 30:        cog_correct += 1.0
    elif hd < 60 and md < 60:      cog_correct += 0.7
    elif h_ang != 0 or m_ang != 0: cog_correct += 0.3

    # Shape
    cog_total += 1
    if cog_answers.get("shapeClicked") == "triangle": cog_correct += 0.5
    if cog_answers.get("largestShape") == "circle":   cog_correct += 0.5
    elif cog_answers.get("largestShape",""):           cog_correct += 0.2

    # Story
    cog_total += 1
    sa = cog_answers.get("storyAnswers", [])
    cog_correct += (
        (1 if len(sa) > 0 and sa[0] == "market" else 0) +
        (1 if len(sa) > 1 and sa[1] == "three"  else 0) +
        (1 if len(sa) > 2 and sa[2] == "bus"    else 0)
    ) / 3.0

    # If nothing attempted — maximum cognitive risk
    attempted = sum([
        1 if cog_answers.get("dayOfWeek") else 0,
        1 if cog_answers.get("currentYear") else 0,
        1 if str(cog_answers.get("location","")).strip() else 0,
        1 if str(cog_answers.get("mathSpent","")).strip() else 0,
        1 if len(cog_answers.get("animals",[])) > 0 else 0,
        1 if any(str(r).strip() for r in cog_answers.get("objectRecall",[])) else 0,
        1 if (cog_answers.get("clockHourAngle",0) != 0) else 0,
        1 if cog_answers.get("shapeClicked") else 0,
        1 if cog_answers.get("largestShape") else 0,
        1 if any(cog_answers.get("storyAnswers",[])) else 0,
    ])
    if attempted == 0:
        cog_pct = 0.0
    else:
        cog_pct = (cog_correct / max(cog_total, 1)) * 100
 
    # ── Speech score: derived from ML probability OR word count override ────────
    # speechRisk 0-100 where 0=healthy, 100=high-risk
    if speech_risk_override is not None:
        speech_risk = speech_risk_override   # word-count based override
    else:
        speech_risk = round(p_dem * 100, 1)  # trust the ML model
    cognitive_risk  = round(100 - cog_pct, 1)
 
    # Combined risk (mirrors original formula inverted):
    #  original: overall = 100 - (speechScore*0.3 + cogPercent*0.7)
    #  Now both are already risk values (higher=worse)
    overall_risk = round(speech_risk * 0.4 + cognitive_risk * 0.6, 1)
    overall_risk = min(100.0, max(0.0, overall_risk))
 
    return {
        "speechScore":    round(speech_risk, 1),    # risk 0-100
        "cognitiveScore": round(cognitive_risk, 1),  # risk 0-100
        "overallRisk":    round(overall_risk, 1),
        "riskLevel":      _risk_level(overall_risk),
    }
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 7.  ROUTES
# ──────────────────────────────────────────────────────────────────────────────
 
# ── Health ────────────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": ml_model is not None,
        "threshold": THRESHOLD,
        "features": len(FEATURE_ORDER),
    })
 
# ── Auth ──────────────────────────────────────────────────────────────────────
@app.route("/api/signup", methods=["POST"])
def signup():
    d = request.json or {}
    required = ["fullName","email","password"]
    if not all(d.get(k) for k in required):
        return jsonify({"error": "Full name, email and password are required"}), 400
    if d["password"] != d.get("confirmPassword",""):
        return jsonify({"error": "Passwords do not match"}), 400
    try:
        with _db() as db:
            db.execute(
                "INSERT INTO users(full_name,age,email,phone,caretaker_name,caretaker_phone,password_hash) "
                "VALUES(?,?,?,?,?,?,?)",
                (d["fullName"], d.get("age"), d["email"], d.get("phone"),
                 d.get("caretakerName"), d.get("caretakerPhone"), _hash(d["password"]))
            )
            user = db.execute("SELECT * FROM users WHERE email=?", (d["email"],)).fetchone()
        return jsonify({"token": _make_token(user["id"]), "user": _safe_user(dict(user))})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email already registered"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
 
@app.route("/api/login", methods=["POST"])
def login():
    d = request.json or {}
    with _db() as db:
        user = db.execute("SELECT * FROM users WHERE email=?", (d.get("email",""),)).fetchone()
    if not user or user["password_hash"] != _hash(d.get("password","")):
        return jsonify({"error": "Invalid email or password"}), 401
    return jsonify({"token": _make_token(user["id"]), "user": _safe_user(dict(user))})
 
 
@app.route("/api/me", methods=["GET"])
@auth_required
def me():
    return jsonify({"user": _safe_user(request.current_user)})
 
 
@app.route("/api/profile", methods=["PUT"])
@auth_required
def update_profile():
    d   = request.json or {}
    uid = request.current_user["id"]
    with _db() as db:
        db.execute(
            "UPDATE users SET full_name=?, age=?, phone=?, caretaker_name=?, "
            "caretaker_phone=?, avatar=? WHERE id=?",
            (d.get("fullName"), d.get("age"), d.get("phone"),
             d.get("caretakerName"), d.get("caretakerPhone"),
             d.get("avatar"), uid)
        )
        user = db.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    return jsonify({"user": _safe_user(dict(user))})
 
 
# ── Assessment ────────────────────────────────────────────────────────────────
@app.route("/api/assess", methods=["POST"])
@auth_required
def assess():
    """
    Main endpoint called by the Lovable frontend after the screening test.
 
    Expected JSON body:
    {
      "transcripts": {            ← speech texts from Web Speech API
        "speech1": "...",         (picture description)
        "speech2": "...",         (passage reading)
        "speech3": "..."          (daily routine)
      },
      "cognitiveAnswers": {       ← cognitive Q&A from ScreeningTest.tsx
        "dayOfWeek":     "Monday",
        "currentYear":   "2025",
        "location":      "Hyderabad",
        "mathSpent":     "700",
        "mathRemaining": "300",
        "animals":       ["dog","cat",...],
        "objectRecall":  ["apple","pen",...],
        "clockHourAngle":   330,
        "clockMinuteAngle": 300,
        "shapeClicked":  "triangle",
        "largestShape":  "circle",
        "storyAnswers":  ["market","three","bus"]
      },
      "audioBase64": "..."        ← optional: base64 WAV for real feature extraction
    }
 
    Returns a TestResult object matching the frontend interface:
    {
      "id": "...",
      "date": "...",
      "speechScore": 34.5,        ← speech risk 0-100 (from ML model)
      "cognitiveScore": 28.1,     ← cognitive risk 0-100
      "overallRisk": 30.4,        ← combined risk 0-100
      "riskLevel": "Low",         ← "Low" | "Moderate" | "High"
      "dementiaProb": 0.34,       ← raw P(dementia) from ensemble
      "modelUsed": "ensemble_pkl"
    }
    """
    uid  = request.current_user["id"]
    body = request.json or {}
 
    transcripts    = body.get("transcripts", {})
    cog_answers    = body.get("cognitiveAnswers", {})
    # Accept per-task audio blobs OR single combined blob
    audio_clips    = body.get("audioClips", {})   # {"speech1": "base64...", ...}
    audio_b64      = body.get("audioBase64", "")   # legacy single blob

    # ── Extract acoustic features from best available audio ───────────────────
    acoustic_feats  = {}
    audio_extracted = False
    try:
        import librosa, base64 as b64lib

        # Pick the largest audio clip (most speech content)
        best_audio = None
        best_size  = 0
        for key in ["speech1", "speech2", "speech3"]:
            clip = audio_clips.get(key, "")
            if clip and len(clip) > best_size:
                best_audio = clip
                best_size  = len(clip)
        if not best_audio and audio_b64:
            best_audio = audio_b64

        if best_audio:
            print(f"🎙  Audio received: {len(best_audio)} chars")
            audio_bytes = b64lib.b64decode(best_audio)

            # Try multiple formats — Safari sends mp4, Chrome sends webm
            extracted = False
            for suffix in [".webm", ".mp4", ".ogg", ".wav"]:
                try:
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = tmp.name
                    acoustic_feats  = extract_acoustic_features(tmp_path)
                    audio_extracted = True
                    extracted = True
                    print(f"✅  Features extracted ({suffix}) — "
                          f"pauses={acoustic_feats.get('num_pauses',0):.0f}, "
                          f"speech_rate={acoustic_feats.get('speech_rate',0):.2f}, "
                          f"word_count={acoustic_feats.get('word_count',0):.0f}")
                    try: os.unlink(tmp_path)
                    except: pass
                    break
                except Exception as inner_e:
                    print(f"⚠️  Format {suffix} failed: {inner_e}")
                    try: os.unlink(tmp_path)
                    except: pass

            if not extracted:
                print("⚠️  All audio formats failed — using transcript-only features")
        else:
            print("ℹ️  No audio blob — using transcript-only features")
    except Exception as e:
        print(f"⚠️  Audio handling failed: {e}")
        acoustic_feats = {}

    # ── Linguistic features from combined transcripts ─────────────────────────
    combined_transcript = " ".join([
        transcripts.get("speech1",""),
        transcripts.get("speech2",""),
        transcripts.get("speech3",""),
    ]).strip()

    linguistic_feats = estimate_linguistic_features(combined_transcript)

    # ── Count actual words spoken ─────────────────────────────────────────────
    # Use acoustic word_count if extracted, else count transcript words
    if audio_extracted and acoustic_feats.get("word_count", 0) > 0:
        total_words_spoken = int(acoustic_feats["word_count"])
        print(f"✅  Using acoustic word_count: {total_words_spoken}")
    else:
        total_words_spoken = len(combined_transcript.strip().split()) if combined_transcript.strip() else 0
        print(f"ℹ️  Using transcript word_count: {total_words_spoken}")

    # ── Build feature vector & run ML model ───────────────────────────────────
    feature_vec = build_feature_vector(acoustic_feats, linguistic_feats)
    prediction  = _predict_with_model(feature_vec)

    print(f"🤖  ML Result: P(dementia)={prediction['dementia_probability']:.4f} "
          f"P(control)={prediction['control_probability']:.4f} "
          f"→ {'DEMENTIA' if prediction['prediction']==1 else 'CONTROL'} "
          f"[{prediction['model_used']}]")
    print(f"    Acoustic: pauses={acoustic_feats.get('num_pauses',0):.0f}, "
          f"speech_rate={acoustic_feats.get('speech_rate',0):.2f}, "
          f"pause_ratio={acoustic_feats.get('pause_ratio',0):.3f}")
    print(f"    Linguistic: words={linguistic_feats.get('word_count',0):.0f}, "
          f"ttr={linguistic_feats.get('ttr',0):.3f}, "
          f"repetitions={linguistic_feats.get('repetition_ratio',0):.3f}")

    # ── Compute final scores ──────────────────────────────────────────────────
    scores = _compute_scores(prediction, cog_answers, total_words_spoken)
 
    # ── Build response matching frontend TestResult interface ─────────────────
    assessment_id = str(int(time.time() * 1000))
    result = {
        "id":             assessment_id,
        "date":           datetime.now().isoformat(),
        "speechScore":    scores["speechScore"],
        "cognitiveScore": scores["cognitiveScore"],
        "overallRisk":    scores["overallRisk"],
        "riskLevel":      scores["riskLevel"],
        "dementiaProb":   round(prediction["dementia_probability"], 4),
        "modelUsed":      prediction["model_used"],
    }
 
    # ── Persist to DB ─────────────────────────────────────────────────────────
    feats_summary = {k: round(v, 4) for k, v in linguistic_feats.items()}
    feats_summary.update({
        "num_pauses":     acoustic_feats.get("num_pauses", 0),
        "pause_ratio":    round(acoustic_feats.get("pause_ratio", 0), 4),
        "speech_rate":    round(acoustic_feats.get("speech_rate", 0), 4),
        "tempo":          round(acoustic_feats.get("tempo", 0), 4),
        "dementia_prob":  result["dementiaProb"],
    })
    with _db() as db:
        db.execute(
            "INSERT INTO assessments(user_id,speech_score,cognitive_score,overall_risk,"
            "risk_level,dementia_prob,model_used,features_json,report_json) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (uid, scores["speechScore"], scores["cognitiveScore"], scores["overallRisk"],
             scores["riskLevel"], prediction["dementia_probability"],
             prediction["model_used"], json.dumps(feats_summary), json.dumps(result))
        )
 
    return jsonify(result)
 
 
# ── History ───────────────────────────────────────────────────────────────────
@app.route("/api/history", methods=["GET"])
@auth_required
def history():
    uid = request.current_user["id"]
    with _db() as db:
        rows = db.execute(
            "SELECT id,taken_at,speech_score,cognitive_score,overall_risk,risk_level,dementia_prob,model_used "
            "FROM assessments WHERE user_id=? ORDER BY taken_at DESC",
            (uid,)
        ).fetchall()
    return jsonify({
        "history": [
            {
                "id":             str(r["id"]),
                "date":           r["taken_at"],
                "speechScore":    r["speech_score"],
                "cognitiveScore": r["cognitive_score"],
                "overallRisk":    r["overall_risk"],
                "riskLevel":      r["risk_level"],
                "dementiaProb":   r["dementia_prob"],
                "modelUsed":      r["model_used"],
            }
            for r in rows
        ]
    })
 
 
@app.route("/api/history/<int:aid>", methods=["GET"])
@auth_required
def get_assessment(aid):
    uid = request.current_user["id"]
    with _db() as db:
        row = db.execute(
            "SELECT * FROM assessments WHERE id=? AND user_id=?", (aid, uid)
        ).fetchone()
    if not row:
        return jsonify({"error": "Not found"}), 404
    r = dict(row)
    r["report_json"]  = json.loads(r.get("report_json") or "{}")
    r["features_json"] = json.loads(r.get("features_json") or "{}")
    return jsonify(r)
 
 
# ── Nearby doctors ────────────────────────────────────────────────────────────
@app.route("/api/nearby-doctors", methods=["POST"])
@auth_required
def nearby_doctors():
    loc = (request.json or {}).get("location", "your city")
    doctors = [
        {"name":"Dr. Priya Sharma",  "speciality":"Neurologist",  "hospital":"Apollo Hospital",  "address":f"Banjara Hills, {loc}",  "phone":"040-23607777","rating":4.8,"distance":"2.1 km","maps_url":f"https://www.google.com/maps/search/neurologist+{loc}"},
        {"name":"Dr. Rajesh Kumar",  "speciality":"Geriatrician", "hospital":"KIMS Hospital",    "address":f"Secunderabad, {loc}",   "phone":"040-44885000","rating":4.7,"distance":"3.4 km","maps_url":f"https://www.google.com/maps/search/geriatrician+{loc}"},
        {"name":"Dr. Anita Reddy",   "speciality":"Neurologist",  "hospital":"Yashoda Hospital", "address":f"Somajiguda, {loc}",     "phone":"040-45670000","rating":4.6,"distance":"4.2 km","maps_url":f"https://www.google.com/maps/search/neurologist+{loc}"},
        {"name":"Dr. Suresh Menon",  "speciality":"Psychiatrist", "hospital":"Care Hospital",    "address":f"Jubilee Hills, {loc}",  "phone":"040-30418000","rating":4.5,"distance":"5.0 km","maps_url":f"https://www.google.com/maps/search/psychiatrist+{loc}"},
    ]
    return jsonify({"doctors": doctors, "location": loc})
 
 
# ── Catch-all OPTIONS for CORS preflight ─────────────────────────────────────
@app.route("/api/<path:path>", methods=["OPTIONS"])
def options_handler(path):
    return jsonify({}), 200
 
 
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  MindGuard Backend  →  http://localhost:5050")
    print(f"  Model loaded       →  {ml_model is not None}")
    print(f"  Threshold          →  {THRESHOLD}")
    print("=" * 55)
    app.run(debug=True, port=5050, host="0.0.0.0")