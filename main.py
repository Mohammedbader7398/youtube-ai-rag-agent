import json
import os
import re
import urllib.parse
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pydantic import BaseModel
import requests
from requests import Session
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DOCTOR_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODEL = "llama-3.3-70b-versatile"
HISTORY_FILE = "video_history.json"

app = FastAPI(title="YouTube RAG Agent")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


class ProcessRequest(BaseModel):
  url: str


class ChatRequest(BaseModel):
  video_id: str
  question: str


def load_history() -> list[dict]:
  if os.path.exists(HISTORY_FILE):
    try:
      with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
    except Exception:
      return []
  return []


def save_to_history(video_meta: dict):
  history = load_history()
  history = [h for h in history if h.get("video_id") != video_meta["video_id"]]
  history.insert(0, video_meta)
  with open(HISTORY_FILE, "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)


def extract_video_id(url: str) -> str:
  pattern = (
      r'(?:v=|\/|youtu\.be\/|embed\/|shorts\/|\bvi?=|\bwatch\?v=)([0-9A-Za-z_-]{11})'
  )
  match = re.search(pattern, url)
  if match:
    return match.group(1)
  raise ValueError("رابط يوتيوب غير صالح")


def fetch_video_details(video_id: str) -> dict:
  url = f"https://www.youtube.com/watch?v={video_id}"
  oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
  try:
    res = requests.get(oembed_url, timeout=5)
    if res.status_code == 200:
      data = res.json()
      return {
          "title": data.get("title", f"فيديو {video_id}"),
          "author": data.get("author_name", "قناة غير معروفة"),
          "thumbnail": (
              data.get("thumbnail_url")
              or f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
          ),
      }
  except Exception:
    pass
  return {
      "title": f"فيديو {video_id}",
      "author": "يوتيوب",
      "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
  }


def format_timestamp(seconds: float) -> str:
  total_seconds = int(seconds)
  hours, remainder = divmod(total_seconds, 3600)
  minutes, secs = divmod(remainder, 60)
  return (
      f"{hours:02d}:{minutes:02d}:{secs:02d}"
      if hours > 0
      else f"{minutes:02d}:{secs:02d}"
  )


def fetch_transcript(video_id: str):
  session = Session()
  session.headers.update({
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
          " (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
      ),
      "Accept-Language": "ar,en-US;q=0.9,en;q=0.8",
  })
  try:
    ytt = (
        YouTubeTranscriptApi(http_client=session)
        if callable(YouTubeTranscriptApi)
        else YouTubeTranscriptApi
    )
  except Exception:
    ytt = (
        YouTubeTranscriptApi()
        if callable(YouTubeTranscriptApi)
        else YouTubeTranscriptApi
    )

  try:
    t_list = (
        ytt.list(video_id)
        if hasattr(ytt, "list")
        else YouTubeTranscriptApi.list_transcripts(video_id)
    )
    target_tr = None
    try:
      target_tr = t_list.find_transcript(["ar", "en"])
    except Exception:
      pass
    if not target_tr:
      try:
        target_tr = t_list.find_generated_transcript(["ar", "en"])
      except Exception:
        pass
    if not target_tr:
      target_tr = next(iter(t_list))

    fetched = target_tr.fetch()
    return (
        fetched.to_raw_data()
        if hasattr(fetched, "to_raw_data")
        else list(fetched)
    )
  except Exception as e:
    err_msg = str(e)
    if (
        "blocking requests from your IP" in err_msg
        or "RequestBlocked" in err_msg
    ):
      raise RuntimeError(
          "يوتيوب حظر الـ IP مؤقتاً. جرب فصل وإعادة توصيل الإنترنت لتغيير الـ IP."
      )
    raise RuntimeError(f"تعذر استخراج النص: {err_msg}")


def chunk_data(transcript_items, max_chars=700):
  chunks, curr_items, curr_len = [], [], 0
  for item in transcript_items:
    t = (item["text"] if isinstance(item, dict) else item.text).replace(
        "\n", " "
    ).strip()
    s = float(item["start"] if isinstance(item, dict) else item.start)
    curr_items.append({"text": t, "start": s})
    curr_len += len(t) + 1

    if curr_len >= max_chars:
      chunk_text = " ".join([x["text"] for x in curr_items])
      chunk_start = curr_items[0]["start"]
      chunks.append({
          "text": chunk_text,
          "start_time": chunk_start,
          "timestamp": format_timestamp(chunk_start),
      })
      curr_items = curr_items[-2:] if len(curr_items) >= 2 else []
      curr_len = sum(len(x["text"]) + 1 for x in curr_items)

  if curr_items:
    chunks.append({
        "text": " ".join([x["text"] for x in curr_items]),
        "start_time": curr_items[0]["start"],
        "timestamp": format_timestamp(curr_items[0]["start"]),
    })
  return chunks


def generate_full_analysis(
    title: str, author: str, chunks: list[dict]
) -> tuple[str, list[dict]]:
  """توليد تحليل شامل لكامل محتوى الفيديو عبر تجميع عينات من مختلف أوقات الفيديو."""
  # اختيار عينات تمثل بداية ووسط ونهاية الفيديو
  step = max(1, len(chunks) // 8)
  sample_chunks = chunks[::step][:8]

  overview_text = "\n\n".join(
      [f"[{c['timestamp']}]: {c['text']}" for c in sample_chunks]
  )

  system_prompt = (
      "أنت محلل محتوى فيديو احترافي. مهمتك تقديم تقرير تحليلي شامل ومنسق"
      " بالماركداون (Markdown) لكامل الفيديو."
  )
  user_prompt = f"""إليك مقاطع ممثلة لكافة أوقات الفيديو:
عنوان الفيديو: "{title}"
صاحب القناة: "{author}"

نصوص من مختلف فترات الفيديو:
{overview_text}

المطلوب: قدّم تحليلاً شاملاً للمحتوى يحتوي على:
1. 🎯 **الملخص التنفيذي**: فكرة الفيديو العامة في فقرة مركزة.
2. 📌 **المحاور الرئيسية**: أهم الأفكار والدروس المذكورة.
3. ⏱️ **الفهرس الزمني وأهم المحطات**: جدول أو نقاط لأهم التوقيتات المذكورة.
4. 💡 **الخلاصة والقيمة العملية**: ماذا يستفيد المشاهد بعد متابعة هذا المقطع؟"""

  try:
    resp = groq_client.chat.completions.create(
        model=DOCTOR_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=900,
    )
    return resp.choices[0].message.content, sample_chunks
  except Exception:
    try:
      resp = groq_client.chat.completions.create(
          model=FALLBACK_MODEL,
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt},
          ],
          temperature=0.3,
          max_tokens=900,
      )
      return resp.choices[0].message.content, sample_chunks
    except Exception as e:
      return f"تم تحليل الفيديو وتخزين {len(chunks)} مقطع بنجاح.", sample_chunks


@app.get("/")
def serve_home():
  return FileResponse("static/index.html")


@app.get("/api/history")
def get_history():
  return load_history()


# مسار البحث المباشر في يوتيوب
@app.get("/api/search")
def search_youtube(q: str):
  if not q.strip():
    return []
  try:
    encoded = urllib.parse.quote_plus(q.strip())
    url = f"https://www.youtube.com/results?search_query={encoded}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ar,en-US;q=0.9,en;q=0.8",
    }
    res = requests.get(url, headers=headers, timeout=6)
    vids = []
    for vid in re.findall(r"/watch\?v=([0-9A-Za-z_-]{11})", res.text):
      if vid not in vids:
        vids.append(vid)
      if len(vids) >= 6:
        break

    results = []
    for vid in vids:
      meta = fetch_video_details(vid)
      results.append({
          "video_id": vid,
          "url": f"https://www.youtube.com/watch?v={vid}",
          **meta,
      })
    return results
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process-video")
def process_video(req: ProcessRequest):
  try:
    video_id = extract_video_id(req.url)
    details = fetch_video_details(video_id)
    coll_name = f"video_{video_id.replace('-', '_')}"

    history = load_history()
    existing = next((h for h in history if h.get("video_id") == video_id), None)

    # إذا كان الفيديو محللاً سابقاً وله تحليل جاهز، نرجعه فوراً
    if existing and existing.get("full_analysis"):
      save_to_history(existing)
      return {"status": "success", **existing}

    raw_transcript = fetch_transcript(video_id)
    chunks = chunk_data(raw_transcript)

    try:
      chroma_client.delete_collection(name=coll_name)
    except Exception:
      pass

    collection = chroma_client.create_collection(
        name=coll_name, embedding_function=embedding_func
    )
    collection.add(
        documents=[c["text"] for c in chunks],
        metadatas=[
            {"start_time": c["start_time"], "timestamp": c["timestamp"]}
            for c in chunks
        ],
        ids=[f"{video_id}_{i}" for i in range(len(chunks))],
    )

    # توليد التقرير التحليلي الشامل لكامل محتوى الفيديو
    full_analysis, sample_sources = generate_full_analysis(
        details["title"], details["author"], chunks
    )

    record = {
        "video_id": video_id,
        "url": req.url,
        "title": details["title"],
        "author": details["author"],
        "thumbnail": details["thumbnail"],
        "chunks_count": len(chunks),
        "full_analysis": full_analysis,
    }
    save_to_history(record)
    return {"status": "success", **record}
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))

def extract_time_from_query(query: str):
  """استخراج التوقيت بالثواني إذا سأل المستخدم عن دقيقة أو توقيت محدد."""
  # فحص صيغة 10:30
  m = re.search(r"(\d{1,2}):(\d{2})", query)
  if m:
    return int(m.group(1)) * 60 + int(m.group(2))

  # فحص "الدقيقة 10" أو "دقيقة 10" أو "دقيقة 7.5"
  m = re.search(r"(?:الدقيق[ةه]|دقيق[ةه]|د)\s*(\d+(?:\.\d+)?)", query)
  if m:
    return int(float(m.group(1)) * 60)

  # فحص "الثانية 40"
  m = re.search(r"(?:الثاني[ةه]|ثاني[ةه])\s*(\d+)", query)
  if m:
    return int(m.group(1))

  return None

@app.post("/api/chat")
def chat_with_video(req: ChatRequest):
  try:
    coll_name = f"video_{req.video_id.replace('-', '_')}"
    collection = chroma_client.get_collection(
        name=coll_name, embedding_function=embedding_func
    )

    sources = []
    target_sec = extract_time_from_query(req.question)

    # 1. إذا سأل المستخدم عن دقيقة/توقيت معين، نجلب المقاطع الأقرب زمنياً لتلك الدقيقة
    if target_sec is not None:
      all_data = collection.get(include=["documents", "metadatas"])
      docs = all_data["documents"]
      metas = all_data["metadatas"]

      # ترتيب المقاطع حسب قربها الزمني من الدقيقة المطلوبة
      sorted_chunks = sorted(
          zip(docs, metas),
          key=lambda x: abs(float(x[1].get("start_time", 0)) - target_sec),
      )
      # أخذ أقرب 3 مقاطع تدور حول هذا الوقت
      for doc, meta in sorted_chunks[:3]:
        sources.append({
            "text": doc,
            "timestamp": meta["timestamp"],
            "start_time": meta["start_time"],
        })
    else:
      # 2. إذا كان السؤال عن موضوع أو فكرة عامة، نستخدم البحث الدلالي العادي (Semantic Search)
      results = collection.query(query_texts=[req.question], n_results=4)
      for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        sources.append({
            "text": doc,
            "timestamp": meta["timestamp"],
            "start_time": meta["start_time"],
        })

    # استرجاع بيانات الفيديو
    history = load_history()
    video_meta = next(
        (h for h in history if h.get("video_id") == req.video_id), {}
    )
    v_title = video_meta.get("title", "هذا الفيديو")
    v_author = video_meta.get("author", "صاحب القناة")

    context_text = "\n\n".join([
        f"--- [مقطع من دقيقة {s['timestamp']}] ---\n{s['text']}"
        for s in sources
    ])

    system_prompt = f"""أنت مساعد ذكي مخصص حصرياً للإجابة عن أسئلة متعلقة بفيديو يوتيوب المعروض حالياً:
عنوان الفيديو: "{v_title}"
القناة: "{v_author}"

قواعد الإجابة:
1. إذا سأل المستخدم عن ما قيل في دقيقة أو وقت محدد، استخدم المقاطع المرفقة التي تم استخراجها من ذلك التوقيت تحديداً للإجابة عما دار في تلك اللحظة.
2. إذا كانت الرسالة مجرد شكر أو تحية، رد بلطف واختصار في جملة واحدة.
3. إذا كان السؤال خارج موضوع الفيديو تماماً، اعتذر بلطف وأخبره أنك مخصص لهذا الفيديو فقط.
4. اذكر دائماً التوقيت الدقيق الذي تتحدث عنه في إجابتك."""

    user_prompt = f"""المقاطع المستخرجة من الفيديو:
{context_text}

سؤال المستخدم: {req.question}
الإجابة:"""

    try:
      resp = groq_client.chat.completions.create(
          model=DOCTOR_MODEL,
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt},
          ],
          temperature=0.3,
          max_tokens=700,
      )
      answer = resp.choices[0].message.content
    except Exception as groq_err:
      if "model_not_found" in str(groq_err).lower() or "404" in str(groq_err):
        resp = groq_client.chat.completions.create(
            model=FALLBACK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=700,
        )
        answer = resp.choices[0].message.content
      else:
        raise groq_err

    return {"answer": answer, "sources": sources}
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


app.mount("/static", StaticFiles(directory="static"), name="static")