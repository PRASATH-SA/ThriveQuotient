# app.py
import os
import json
import sqlite3
from io import BytesIO
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import matplotlib.pyplot as plt
import requests

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ---------- Config ----------
LOCAL_MODEL_PATH = "./models/all-MiniLM-L6-v2"
HF_MODEL_NAME = "all-MiniLM-L6-v2"
DB_PATH = "ikigai_dynamic.db"
SERPAPI_KEY = os.environ.get("SERPAPI_API_KEY")  # set this env var for live course search

app = Flask(__name__)
app.config['DB_PATH'] = DB_PATH

# ---------- Domain question bank (5 canonical Qs) ----------
DOMAIN_QUESTIONS = {
    "Technology": [
        "Describe a recent tech project or idea that excited you — which part did you enjoy most?",
        "When solving a problem, do you prefer designing systems, coding, or analyzing data?",
        "Would you rather build a user-facing app or optimize backend systems?",
        "Which programming tools or languages do you naturally reach for first?",
        "If you had a weekend to build something, what would it be?"
    ],
    "Medicine": [
        "What areas of healthcare or medicine captivate you most?",
        "Do you prefer clinical practice, lab research, or healthcare systems design?",
        "Which biomedical topics do you read about for fun?",
        "Would you rather work in patient-facing care or medical data/research?",
        "If you could change one thing in healthcare using technology, what would it be?"
    ],
    "Art": [
        "Which visual art forms draw you in (painting, illustration, digital art)?",
        "Describe an artwork you made that made you proud — why?",
        "Do you enjoy generating concepts or perfecting details?",
        "Which tools (brush, tablet, software) do you prefer?",
        "If you had a gallery show, what concept would you explore?"
    ],
    "Music": [
        "Which music activities energize you: composing, performing, producing, or teaching?",
        "Tell me about a musical piece/project you loved working on and why.",
        "Do you focus more on melody, rhythm, lyrics, or sound design?",
        "Which instruments or DAWs do you use or want to learn?",
        "Would you prefer to collaborate or create solo music?"
    ],
    "Science": [
        "Which scientific fields spark your curiosity most?",
        "Do you prefer hands-on experiments, data analysis, or theoretical work?",
        "Describe an experiment or project you'd like to run.",
        "Which scientific tools or software do you enjoy or want to learn?",
        "Would you rather work in lab research, field work, or computational models?"
    ],
    "Business": [
        "Which business area excites you: product, strategy, sales, or finance?",
        "Describe a business idea you'd love to build.",
        "Do you enjoy talking to customers, analyzing markets, or designing models?",
        "Which business tasks do you naturally assume in group work?",
        "If you could launch one product today, what problem would it solve?"
    ],
    "Sports": [
        "Which sports roles excite you most: playing, coaching, conditioning, or analytics?",
        "Tell me about a coaching or sports project you're proud of.",
        "Do you prefer technique work, strategy, or physical conditioning?",
        "Which drills or metrics do you track or enjoy improving?",
        "Would you rather train athletes, design programs, or analyze performance?"
    ],
    "Teaching": [
        "What subjects do you most enjoy explaining to others?",
        "Do you prefer one-on-one mentoring, classroom teaching, or designing courses?",
        "Describe a teaching moment that felt rewarding.",
        "Which audience or age group do you enjoy teaching?",
        "Would you rather make online courses, tutor live, or design curricula?"
    ],
}

# ---------- Candidate labels covering many fields ----------
CANDIDATE_LABELS = [
    "web development","mobile development","backend engineering","frontend engineering","devops","cloud engineering",
    "cybersecurity","robotics","embedded systems","data science","data analytics","machine learning","deep learning",
    "research","ui ux design","graphic design","illustration","photography","video production","animation",
    "music production","songwriting","performance","composition","clinical care","medical research","healthcare analytics",
    "bioinformatics","product management","business strategy","entrepreneurship","finance","sales","marketing",
    "content creation","sports coaching","fitness training","strength conditioning","sports analytics","teaching",
    "instructional design","curriculum development","mentoring"
]

# ---------- Segment prototypes (for mapping clusters to human segments) ----------
SEGMENT_PROTOTYPES = {
    "Technology": ["web","backend","frontend","devops","cloud","embedded","robotics","cybersecurity"],
    "Data & Research": ["data","machine learning","analytics","research","bioinformatics","statistics"],
    "Design & Creative": ["design","ui","ux","graphic","illustration","photography","video","animation"],
    "Music & Audio": ["music","production","songwriting","composition","performance","sound"],
    "Medicine & Healthcare": ["clinical","medical","healthcare","bioinformatics"],
    "Business & Product": ["product","business","entrepreneur","finance","sales","marketing"],
    "Sports & Fitness": ["sports","coaching","fitness","conditioning","analytics"],
    "Teaching & Education": ["teaching","instruction","curriculum","mentoring"]
}

# ---------- Static fallback course links (curated) ----------
FALLBACK_COURSES = {
    "Technology":[ {"title":"Full Stack Web Dev (Udemy)","url":"https://www.udemy.com"}, {"title":"Cloud Engineering (Coursera)","url":"https://www.coursera.org"}, {"title":"Robotics Intro","url":"https://www.coursera.org"} ],
    "Data & Research":[ {"title":"IBM Data Science (Coursera)","url":"https://www.coursera.org"},{"title":"ML (Andrew Ng)","url":"https://www.coursera.org"},{"title":"Deep Learning Specialization","url":"https://www.coursera.org"} ],
    "Design & Creative":[ {"title":"Google UX Design","url":"https://www.coursera.org"},{"title":"Graphic Design Masterclass","url":"https://www.udemy.com"},{"title":"Photography Course","url":"https://www.udemy.com"} ],
    "Music & Audio":[ {"title":"Music Production Specialization","url":"https://www.coursera.org"},{"title":"Ableton Course (Udemy)","url":"https://www.udemy.com"},{"title":"Songwriting (Berklee)","url":"https://www.coursera.org"} ],
    "Medicine & Healthcare":[ {"title":"Healthcare Analytics","url":"https://www.coursera.org"},{"title":"Clinical Research Methods","url":"https://www.coursera.org"},{"title":"Bioinformatics Intro","url":"https://www.coursera.org"} ],
    "Business & Product":[ {"title":"Product Management (Coursera)","url":"https://www.coursera.org"},{"title":"Entrepreneurship Essentials","url":"https://www.edx.org"},{"title":"Business Analytics","url":"https://www.coursera.org"} ],
    "Sports & Fitness":[ {"title":"Sports Coaching","url":"https://www.coursera.org"},{"title":"Fitness Trainer Certification","url":"https://www.coursera.org"},{"title":"Sports Analytics","url":"https://www.coursera.org"} ],
    "Teaching & Education":[ {"title":"Instructional Design","url":"https://www.coursera.org"},{"title":"Teach Online (Coursera)","url":"https://www.coursera.org"},{"title":"TESOL Certificate","url":"https://www.tefl.org"} ],
}

# ---------- DB helpers ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            domains TEXT,
            created_at TEXT,
            answers_json TEXT,
            result_json TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_profile(name, email, domains, answers, result):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('INSERT INTO profiles (name,email,domains,created_at,answers_json,result_json) VALUES (?, ?, ?, ?, ?, ?)',
                (name, email, json.dumps(domains), datetime.utcnow().isoformat(), json.dumps(answers), json.dumps(result)))
    pid = cur.lastrowid
    conn.commit()
    conn.close()
    return pid

def load_profile(pid):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('SELECT id,name,email,domains,created_at,answers_json,result_json FROM profiles WHERE id=?', (pid,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "name": row[1],
        "email": row[2],
        "domains": json.loads(row[3]),
        "created_at": row[4],
        "answers": json.loads(row[5]),
        "result": json.loads(row[6])
    }

init_db()

# ---------- Load embedder (local preferred) ----------
def load_embedder():
    if os.path.isdir(LOCAL_MODEL_PATH):
        print(f"Loading local embedder from {LOCAL_MODEL_PATH} ...")
        return SentenceTransformer(LOCAL_MODEL_PATH)
    else:
        print(f"Local model not found; downloading {HF_MODEL_NAME} ...")
        return SentenceTransformer(HF_MODEL_NAME)

print("Loading embedding model...")
embedder = load_embedder()
print("Embedder ready.")

# ---------- Precompute label embeddings and KMeans ----------
label_embeddings = embedder.encode(CANDIDATE_LABELS, convert_to_numpy=True, show_progress_bar=False)
n_clusters = min(len(SEGMENT_PROTOTYPES), max(3, len(CANDIDATE_LABELS)//6))
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(label_embeddings)
cluster_centroids = kmeans.cluster_centers_

# Map cluster -> human segment via prototype similarity
proto_embeddings = {}
for seg, words in SEGMENT_PROTOTYPES.items():
    proto_embeddings[seg] = np.mean(embedder.encode(words, convert_to_numpy=True, show_progress_bar=False), axis=0)

cluster_to_segment = {}
for i, centroid in enumerate(cluster_centroids):
    best_seg = None
    best_sim = -1.0
    for seg, proto in proto_embeddings.items():
        sim = float(np.dot(centroid / (np.linalg.norm(centroid)+1e-12), proto / (np.linalg.norm(proto)+1e-12)))
        if sim > best_sim:
            best_sim = sim
            best_seg = seg
    cluster_to_segment[i] = best_seg

print("Cluster->Segment map:", cluster_to_segment)

# ---------- Utilities: map answer -> nearest label and segment ----------
def answer_to_label(answer, top_k=1):
    emb = embedder.encode([answer], convert_to_numpy=True)[0]
    sims = cosine_similarity(label_embeddings, emb.reshape(1, -1)).reshape(-1)
    top_idx = sims.argsort()[::-1][:top_k]
    return [(CANDIDATE_LABELS[i], float(sims[i])) for i in top_idx]

def label_to_segment(label):
    try:
        idx = CANDIDATE_LABELS.index(label)
    except ValueError:
        emb = embedder.encode([label], convert_to_numpy=True)[0]
        sims = cosine_similarity(label_embeddings, emb.reshape(1,-1)).reshape(-1)
        idx = int(np.argmax(sims))
    cluster_idx = int(kmeans.labels_[idx])
    return cluster_to_segment.get(cluster_idx, "Business & Product")

# ---------- SerpAPI: fetch top-3 links for a segment ----------
def serpapi_search_courses(segment, num_results=3):
    if not SERPAPI_KEY:
        return None
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": f"best {segment} courses online",
        "num": num_results,
        "hl": "en",
        "api_key": SERPAPI_KEY
    }
    try:
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        results = []
        # prefer organic_results
        org = data.get("organic_results") or data.get("top_results") or []
        for item in org[:num_results]:
            title = item.get("title") or item.get("position") or "Course"
            link = item.get("link") or item.get("displayed_link") or item.get("source")
            if link:
                results.append({"title": title, "url": link})
        # fallback to answer_box
        if not results and "answer_box" in data:
            ab = data["answer_box"]
            if 'links' in ab:
                for l in ab['links'][:num_results]:
                    results.append({"title": l.get("text","Course"), "url": l.get("link")})
        return results[:num_results] if results else None
    except Exception as e:
        print("SerpAPI error:", e)
        return None

# ---------- Adaptive follow-up logic ----------
# If a label contains some keyword, map to a followup question template
FOLLOWUP_TEMPLATES = {
    "data": "When you think about working with data, do you prefer exploring/visualizing data or building predictive models?",
    "machine": "Do you enjoy creating models (algorithms) or cleaning/engineering the data they use?",
    "web": "Do you enjoy designing user experiences or building back-end features?",
    "design": "Do you enjoy research & concept, or executing polished visuals?",
    "music": "Do you prefer composing original material or producing/engineering the sound?",
    "clinical": "Are you more drawn to direct patient care or clinical research/design?",
    "business": "Would you rather focus on customer discovery & product-market fit or financial/business operations?",
    "sports": "Do you prefer hands-on coaching or analyzing performance metrics to improve outcomes?",
    "teaching": "Do you find creating course structure more enjoyable, or delivering interactive sessions?"
}
GENERIC_FOLLOWUP = "Can you give one specific example (project/task) related to what you just described?"

def choose_followup_from_answers(domain, answers_two):
    # combine text and pick top label
    combined = " ".join(answers_two)
    label, sim = answer_to_label(combined, top_k=1)[0]
    # choose keyword by checking label tokens
    token = label.split()[0].lower()
    for key, templ in FOLLOWUP_TEMPLATES.items():
        if key in label.lower() or key == token:
            return templ, label, sim
    # fallback: check synonyms in label
    for key, templ in FOLLOWUP_TEMPLATES.items():
        if key in combined.lower():
            return templ, label, sim
    return GENERIC_FOLLOWUP, label, sim

# ---------- Aggregation into four Ikigai axes ----------
def aggregate_to_segments_and_axes(detailed_map):
    """
    detailed_map: domain -> list of dicts {answer,label,score,segment}
    returns segments list and radar dict with keys Passion, Profession, Vocation, Mission (0..100)
    """
    # Build per-segment passion and profession signals from answers
    segs = {}
    for domain, items in detailed_map.items():
        for it in items:
            seg = it['segment']
            if seg not in segs:
                segs[seg] = {"passion_scores":[], "profession_scores":[], "labels":[]}
            segs[seg]["labels"].append(it['label'])
            # heuristics:
            txt = it['answer'].lower()
            # if contains love/enjoy -> passion
            if any(w in txt for w in ["enjoy","love","excite","passion","energ","like"]):
                segs[seg]["passion_scores"].append(it['score'])
            else:
                segs[seg]["profession_scores"].append(it['score'])
    # convert to list
    segments = []
    for seg, v in segs.items():
        passion = float(np.mean(v['passion_scores'])) if v['passion_scores'] else 0.0
        profession = float(np.mean(v['profession_scores'])) if v['profession_scores'] else 0.0
        freq = len(v['labels'])
        segments.append({"segment":seg, "passion_score":passion, "profession_score":profession, "frequency":freq, "keywords": list(set(v['labels']))})
    # compute axes across all segments
    if not segments:
        radar = {"Passion":10,"Profession":10,"Vocation":10,"Mission":10}
        return segments, radar
    # Passion = mean passion *100
    passion_vals = [s['passion_score'] for s in segments]
    profession_vals = [s['profession_score'] for s in segments]
    Passion = np.mean(passion_vals)*100
    Profession = np.mean(profession_vals)*100
    # Vocation approximated as how many labels relate to "business/product/market" tokens
    market_tokens = ["product","business","entrepreneur","marketing","sales","finance","growth"]
    vocational_scores = []
    for s in segments:
        ktxt = " ".join(s['keywords']).lower()
        score = sum(1 for t in market_tokens if t in ktxt)
        vocational_scores.append(score)
    if vocational_scores:
        Vocation = (sum(vocational_scores)/ (len(vocational_scores)*len(market_tokens))) * 100
    else:
        Vocation = 10.0
    # Mission approximated as overlap between Passion and Vocation normalized
    Mission = ( (Passion/100.0) * (Vocation/100.0) ) * 100
    def clamp(x): return max(0,min(100,float(x)))
    radar = {"Passion":clamp(Passion), "Profession":clamp(Profession), "Vocation":clamp(Vocation*100 if Vocation<=1 else Vocation), "Mission":clamp(Mission)}
    return segments, radar

# ---------- Roadmap (simple generator) ----------
def generate_roadmap(top_segment, current_profession_score):
    target = 80.0
    gap = max(0, target - current_profession_score)
    months = max(3, int(np.ceil(gap/15)*2))
    steps = [
        {"title": f"Set a {months}-month goal", "details": f"Define a concrete role/project in {top_segment} and measurable outcomes."},
        {"title": "Foundational Learning", "details": "Take 1-2 structured courses and finish hands-on labs."},
        {"title": "Build Portfolio Projects", "details": "Create small and medium projects, publish code/artefacts."},
        {"title": "Practice & Feedback", "details": "Get reviews and iterate on projects."},
        {"title": "Apply & Network", "details": "Start applying and reach out to domain communities."}
    ]
    for s in steps:
        s['duration'] = f"{max(1, months//3)} month(s)"
    return {"target_skill":target, "current_skill":round(current_profession_score,1), "months_estimate":months, "steps":steps}

# ---------- Radar PNG creation ----------
def make_radar_png(radar):
    labels = list(radar.keys())
    values = [radar[k] for k in labels]
    arr = np.array(values)
    arr_closed = np.concatenate([arr, arr[:1]])
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, arr_closed, linewidth=2)
    ax.fill(angles, arr_closed, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0,100)
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf

# ---------- Routes ----------

@app.route("/")
def index():
    domains = ["Technology", "Medicine", "Art", "Music", "Science", "Business"]
    left_facts = [
        "Ikigai means 'reason for being' in Japanese.",
        "It combines passion, mission, vocation, and profession."
    ]
    right_facts = [
        "A balance of all four leads to fulfillment.",
        "Your Ikigai evolves with your experiences."
    ]
    
    DOMAIN_QUESTIONS = {
        "Technology": ["What excites you in programming?", "Which tech problem do you wish to solve?"],
        "Medicine": ["Why do you want to heal people?", "What inspires you about medicine?"],
        "Art": ["What kind of art do you love?", "Do you create for yourself or for others?"],
        "Music": ["What role does music play in your life?", "Do you enjoy performing or composing?"],
        "Science": ["Which scientific discovery inspires you?", "What experiment would you love to try?"],
        "Business": ["What business idea excites you most?", "Do you enjoy solving market problems?"]
    }
    
    return render_template(
        "index.html",
        domains=domains,
        left_facts=left_facts,
        right_facts=right_facts,
        DOMAIN_QUESTIONS=DOMAIN_QUESTIONS  # <-- pass it here
    )

@app.route("/start_session", methods=["POST"])
def start_session():
    data = request.json
    domains = data.get("domains", [])
    if not domains or len(domains) == 0:
        return jsonify({"error":"Select at least one domain."}), 400
    if len(domains) > 2:
        return jsonify({"error":"Max 2 domains allowed."}), 400
    # For each domain we will send first 2 questions only initially (client will call followup)
    questions = []
    for d in domains:
        qs = DOMAIN_QUESTIONS.get(d, [])[:2]  # first two
        for q in qs:
            questions.append({"domain": d, "prompt": q})
    return jsonify({"questions": questions})

@app.route("/followup", methods=["POST"])
def followup():
    """
    expects: { domain: "Technology", answers_two: ["...","..."] }
    returns: { followup: "...", label: "...", sim: float }
    """
    data = request.json
    domain = data.get("domain")
    answers_two = data.get("answers_two", [])
    if not domain or not answers_two or len(answers_two) < 1:
        return jsonify({"error":"Provide domain and answers_two"}), 400
    followup_q, label, sim = choose_followup_from_answers(domain, answers_two)
    return jsonify({"followup": followup_q, "label": label, "sim": sim})

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    payload:
    {
      name, email,
      domains: [...],
      answers: { "Technology": ["ans1","ans2","followup","ans3","ans4","ans5"], ... }
    }
    """
    payload = request.json
    name = payload.get("name","").strip()
    email = payload.get("email","").strip()
    domains = payload.get("domains", [])
    answers = payload.get("answers", {})

    if not domains or not answers:
        return jsonify({"error":"Missing domains or answers"}), 400

    # Build detailed map domain -> list of dicts
    detailed = {}
    for d in domains:
        ans_list = answers.get(d, [])
        detailed[d] = []
        for a in ans_list:
            if not a or not a.strip():
                continue
            label, score = answer_to_label(a, top_k=1)[0]
            seg = label_to_segment(label)
            detailed[d].append({"answer":a, "label":label, "score":score, "segment":seg})

    # Flatten and aggregate
    segments, radar = aggregate_to_segments_and_axes(detailed)
    top_segment = segments[0]['segment'] if segments else "Business & Product"
    current_prof = segments[0]['profession_score']*100 if segments else 0.0
    roadmap = generate_roadmap(top_segment, current_prof)

    # Get live courses via SerpAPI
    courses = serpapi_search_courses(top_segment)
    if not courses:
        # fallback to curated mapping, try direct key match
        courses = FALLBACK_COURSES.get(top_segment, FALLBACK_COURSES.get("Business & Product"))

    result = {"detailed":detailed, "segments":segments, "radar":radar, "top_segment":top_segment, "roadmap":roadmap, "courses":courses}
    pid = save_profile(name, email, domains, detailed, result)
    return jsonify({"profile_id": pid, "result": result})

@app.route("/profile/<int:pid>")
def profile(pid):
    p = load_profile(pid)
    if not p:
        return "Not found", 404
    return render_template("result.html", profile=p)

@app.route("/download_pdf/<int:pid>")
def download_pdf(pid):
    p = load_profile(pid)
    if not p:
        return "Not found", 404
    radar_buf = make_radar_png(p['result']['radar'])
    pdf_buf = BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=A4)
    w, h = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h-60, f"Ikigai Roadmap — {p['name'] or 'Anonymous'}")
    c.setFont("Helvetica", 9)
    c.drawString(40, h-78, f"Domains: {', '.join(p['domains'])}   Created: {p['created_at']}")
    img = ImageReader(radar_buf)
    img_w = 360; img_h = 240
    c.drawImage(img, 40, h-78-img_h-10, width=img_w, height=img_h)
    x = 420; y = h-120
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Top segment:")
    c.setFont("Helvetica", 10)
    c.drawString(x, y-16, p['result']['top_segment'])
    y -= 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Recommended Courses:")
    y -= 18
    c.setFont("Helvetica", 10)
    for cinfo in p['result']['courses']:
        c.drawString(50, y, f"- {cinfo['title']}")
        y -= 14
        if y < 60:
            c.showPage()
            y = h-60
    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Roadmap (summary):")
    y -= 18
    c.setFont("Helvetica", 10)
    for step in p['result']['roadmap']['steps']:
        c.drawString(50, y, f"• {step['title']}: {step['details'][:100]}")
        y -= 14
        if y < 60:
            c.showPage()
            y = h-60
    c.showPage()
    c.save()
    pdf_buf.seek(0)
    filename = f"ikigai_roadmap_{pid}.pdf"
    return send_file(pdf_buf, as_attachment=True, download_name=filename, mimetype='application/pdf')

if __name__ == "__main__":
    print("App starting at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
