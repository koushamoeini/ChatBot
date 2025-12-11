# backend/chatbot.py
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import os
from typing import List, Tuple, Dict, Any, Optional
from chromadb.errors import NotFoundError

class RAGChatbot:
    def __init__(self, api_key: str, base_url: str = "https://api.tapsage.com/openai/v1", relevance_threshold: float = 0.6, low_confidence_threshold: float = 0.4, minimal_sim_threshold: float = 0.15):
        # Embedding model
        self.embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        
        # OpenAI client
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = "gpt-4o-mini"

        # Relevance threshold for semantic similarity filtering (0..1 cosine similarity)
        self.relevance_threshold = relevance_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.minimal_sim_threshold = minimal_sim_threshold

        # ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        try:
            self.collection = self.chroma_client.get_collection(name="farsi_rag_collection")
        except NotFoundError:
            # Ensure the collection exists before querying
            self.collection = self.chroma_client.create_collection(name="farsi_rag_collection")

    def rewrite_query(self, current_query: str, history: List[Dict[str, str]]) -> str:
        if not history:
            return current_query

        history_str = "\n".join([
            f"{'کاربر' if turn['role'] == 'user' else 'پشتیبان'}: {turn['content']}"
            for turn in history[-4:]
        ])

        rewrite_prompt = f"""
        شما یک دستیار هوشمند هستید که وظیفه دارد سوال فعلی کاربر را با توجه به تاریخچه گفتگو، به یک سوال کاملاً مستقل و واضح تبدیل کند.
        - اگر سوال فعلی کامل و مستقل است، همان را بدون تغییر برگردانید.
        - در غیر این صورت، آن را طوری بازنویسی کنید که بدون نیاز به تاریخچه قابل فهم باشد.
        - فقط سوال بازنویسی‌شده را در یک خط و بدون هیچ توضیح اضافه بنویسید.

        تاریخچه گفتگو:
        {history_str}

        سوال فعلی: "{current_query}"
        """

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "شما یک بازنویس‌کننده هوشمند سوالات فارسی هستید."},
                    {"role": "user", "content": rewrite_prompt}
                ],
                temperature=0.0,
                max_tokens=60
            )
            return completion.choices[0].message.content.strip()
        except:
            return current_query

    def summarize_chunks(self, chunks: List[Tuple[str, Dict[str, Any]]]) -> str:
        """
        Produce a short Farsi summary of the retrieved chunks to guide the agent.
        """
        if not chunks:
            return ""
        # Build a compact context to summarize
        parts = []
        for i, (text, meta) in enumerate(chunks, 1):
            title = meta.get("source_title", "منبع نامشخص")
            snippet = text if len(text) < 300 else text[:300] + "..."
            parts.append(f"منبع {i}: {title} — {snippet}")

        prompt = (
            "شما یک خلاصه‌ساز دقیق هستید. لطفاً خلاصه‌ای کوتاه و مفید به زبان فارسی از متن‌های زیر تهیه کنید؛ "
            "حداکثر 2-3 جمله و تمرکز روی نکات کلیدی که می‌تواند به پاسخ‌دهی کمک کند. \n\n"
            + "\n\n".join(parts)
        )

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "شما یک خلاصه‌ساز دقیق متن فارسی هستید."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=150
            )
            return completion.choices[0].message.content.strip()
        except Exception:
            return ""

    def expand_query(self, query: str) -> List[str]:
        correction_prompt = f"""
        شما یک ویرایشگر زبان فارسی هستید. وظیفه شما این است که فقط و فقط اشتباهات املایی، نگارشی یا دستوری سوال زیر را اصلاح کنید.
        - اگر به صورت رسمی نوشته نشده آن را به صورت رسمی بازنویسی کنید.
        - اگر سوال از نظر زبانی صحیح است، همان سوال را بدون هیچ تغییری برگردانید.
        - هیچ بازنویسی معنادار، تغییر سبک یا جایگزینی واژه انجام ندهید.
        - فقط یک خط پاسخ بدهید و هیچ توضیح اضافه ننویسید.

        سوال: "{query}"
        """

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "شما یک ویرایشگر دقیق زبان فارسی هستید."},
                    {"role": "user", "content": correction_prompt}
                ],
                temperature=0.0,
            )
            corrected = completion.choices[0].message.content.strip()
            if not corrected:
                corrected = query
            return [query, corrected]
        except:
            return [query]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        a = np.array(a)
        b = np.array(b)
        if a.size == 0 or b.size == 0:
            return 0.0
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def is_greeting(self, text: str) -> bool:
        """Return True if the input text looks like a greeting in Farsi."""
        if not text or not isinstance(text, str):
            return False
        txt = text.strip().lower()
        greetings = [
            'سلام', 'درود', 'خوش آمدید', 'صبح بخیر', 'عصر بخیر', 'شب بخیر', 'حال شما', 'حالت چطوره', 'خوبی'
        ]
        for g in greetings:
            if g in txt:
                return True
        return False

    def retrieve_relevant_chunks(self, queries: List[str], n_results: int = 6, apply_threshold: bool = True) -> List[Tuple[str, Dict[str, Any]]]:
        seen = set()
        results = []
        for q in queries:
            emb = self.embedding_model.encode(q).tolist()
            res = self.collection.query(query_embeddings=[emb], n_results=n_results)
            docs = res.get('documents', [[]])[0]
            metas = res.get('metadatas', [[]])[0]
            for doc, meta in zip(docs, metas):
                if doc in seen:
                    continue
                # compute similarity between query embedding and the document text embedding
                try:
                    doc_emb = self.embedding_model.encode(doc).tolist()
                    sim = self._cosine_similarity(emb, doc_emb)
                except Exception:
                    sim = 0.0
                if (not apply_threshold) or (sim >= self.relevance_threshold):
                    # annotate meta with computed similarity for downstream processing/inspection
                    meta = dict(meta) if isinstance(meta, dict) else {"source": str(meta)}
                    meta["_similarity"] = sim
                    seen.add(doc)
                    results.append((doc, meta))
        return results

    def is_answer_supported(self, answer: str, chunks: List[Tuple[str, Dict[str, Any]]]) -> Tuple[bool, str]:
        """
        Ask the model to validate whether the `answer` is supported by the `chunks`.
        Returns (supported_bool, reason_str).
        """
        if not chunks:
            return False, "هیچ زمینه‌ای برای ارزیابی وجود ندارد."

        context_parts = []
        for i, (text, meta) in enumerate(chunks, 1):
            title = meta.get("source_title", "منبع نامشخص") if isinstance(meta, dict) else "منبع"
            snippet = text if len(text) < 300 else text[:300] + "..."
            context_parts.append(f"منبع {i}: {title} — {snippet}")

        prompt = (
            "شما یک ارزیاب اطلاعات هستید. تصمیم بگیرید که آیا پاسخ زیر به صورت مستقیم و به‌طور معتبری از متن‌های ارائه‌شده پشتیبانی می‌شود یا خیر. "
            "فقط جواب 'SUPPORTED' یا 'NOT_SUPPORTED' را در یک شیٔ JSON تک‌خطی همراه با یک دلیل کوتاه (Farsi) بدهید.\n\n"
            f"سوال: {answer}\n\n" + "\n\n".join(context_parts)
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "شما یک ارزیاب دقیق هستید که بررسی می‌کند آیا پاسخ بر اساس متن‌های ارائه شده پشتیبانی می‌شود یا خیر."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=100,
            )
            txt = response.choices[0].message.content.strip()
            import json
            parsed = None
            try:
                parsed = json.loads(txt)
            except Exception:
                # fallback parse: look for SUPPORTED or NOT_SUPPORTED
                if 'SUPPORTED' in txt.upper():
                    return True, 'پاسخ با زمینه‌ها مطابقت دارد.'
                return False, txt

            result = parsed if isinstance(parsed, dict) else {}
            status = result.get('status') or result.get('supported') or result.get('SUPPORTED')
            reason = result.get('reason') or result.get('explanation') or ''
            if isinstance(status, bool):
                return status, reason
            if isinstance(status, str) and status.strip().upper().startswith('S'):
                return True, reason
            return False, reason
        except Exception:
            return False, "خطا در ارزیابی پشتیبانی پاسخ از متن‌ها."

    def run_agent(self, question: str, history: List[Dict[str, str]], max_steps: int = 4) -> str:
        """
        A simple agentic ReAct-style loop that lets the model call tools (retrieve/summarize) and then return a final answer.

        Action format expected from LLM (in a single JSON object in a single line):
        {"action": "retrieve", "input": "<query>"}
        {"action": "summarize", "input": "<summary context>"}
        {"action": "final", "input": "<final answer>"}
        """
        # Prepare initial query
        standalone = self.rewrite_query(question, history)
        expanded = self.expand_query(standalone)

        system_prompt = (
            "شما یک عامل (Agent) به زبان فارسی هستید که از ابزارهای زیر استفاده می‌کند:\n"
            "TOOLS:\n"
            "- retrieve(query): دریافت بخش‌های مرتبط از پایگاه دانش\n"
            "- summarize(chunks): خلاصه کردن بخش‌های دریافت‌شده\n"
            "فرمت پاسخ‌ها باید یک شیٔ JSON تک‌خطی با کلیدهای 'action' و 'input' یا 'final' باشد.\n"
            "اگر آمادهٔ ارائه پاسخ نهایی هستید، از action='final' استفاده کنید و پاسخ را در 'input' قرار دهید.\n"
            "همیشه به زبان فارسی پاسخ بدهید."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"سوال اصلی: {question}\nتوضیح: اگر لازم است از ابزارها استفاده کن."}
        ]

        # If the question is a greeting, reply immediately with a short friendly message.
        if self.is_greeting(question):
            return "سلام! خوش آمدید — چطور می‌تونم کمکتون کنم؟"

        tool_outputs: List[str] = []
        chunks_cache: Optional[List[Tuple[str, Dict[str, Any]]]] = None

        for step in range(max_steps):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=400,
                )
                response_text = completion.choices[0].message.content.strip()
            except Exception as e:
                return "یک خطا هنگام ارتباط با مدل رخ داد، لطفا بعدا تلاش کنید."

            # Try parse JSON
            action = None
            payload = None
            try:
                import json as _json
                parsed = _json.loads(response_text)
                action = parsed.get("action")
                payload = parsed.get("input")
            except Exception:
                # If parsing fails, assume final answer is present as plain text
                return response_text

            if action == "retrieve":
                # Use the agent's retrieve tool
                if not payload:
                    messages.append({"role": "assistant", "content": "Action 'retrieve' requires 'input'."})
                    continue
                q_for_retrieve = payload
                res_chunks = self.retrieve_relevant_chunks([q_for_retrieve])
                chunks_cache = res_chunks
                if not res_chunks:
                    # If retrieval doesn't find relevant chunks under strict threshold, try relaxed retrieval
                    relaxed = self.retrieve_relevant_chunks([q_for_retrieve], apply_threshold=False, n_results=2)
                    if not relaxed:
                        # no matches at all
                        return "متاسفانه اطلاعات کافی برای پاسخ به سوال شما در پایگاه دانش ما وجود ندارد."
                    # choose highest similarity among relaxed results
                    top_sim = max([m.get('_similarity', 0.0) for _, m in relaxed if isinstance(m, dict)] or [0.0])
                    if top_sim < self.minimal_sim_threshold:
                        return "متاسفانه اطلاعات کافی برای پاسخ به سوال شما در پایگاه دانش ما وجود ندارد."
                    # use relaxed chunks but mark them low-confidence
                    chunks_cache = relaxed
                    for _, m in chunks_cache:
                        if isinstance(m, dict):
                            m['_low_confidence'] = True
                # Append tool output to messages
                output_text = "\n\n".join([f"[{i+1}] {t} (meta: {m})" for i, (t, m) in enumerate(res_chunks)])
                tool_outputs.append(output_text)
                messages.append({"role": "assistant", "content": f"TOOL_OUTPUT_RETRIEVE: {output_text}"})
                continue

            if action == "summarize":
                if chunks_cache is None:
                    messages.append({"role": "assistant", "content": "هیچ بخشی برای خلاصه کردن موجود نیست. ابتدا retrieve را اجرا کنید."})
                    continue
                summary = self.summarize_chunks(chunks_cache)
                tool_outputs.append(summary)
                messages.append({"role": "assistant", "content": f"TOOL_OUTPUT_SUMMARY: {summary}"})
                continue

            if action == "final":
                # Use the provided final answer
                answer_text = str(payload or "")
                if not answer_text:
                    # If empty, try to generate a final answer using cached chunks
                    try:
                        candidate = self.generate_response(question, chunks_cache or [], history)
                    except Exception:
                        return "متاسفانه پاسخی تولید نشد."
                    # validate generated candidate
                    supported, reason = self.is_answer_supported(candidate, chunks_cache or [])
                    if supported:
                        return candidate
                    else:
                        # If the agent retrieved only low-confidence matches, provide a best-effort answer prefaced
                        if chunks_cache and any([m.get('_low_confidence') for _, m in chunks_cache if isinstance(m, dict)]):
                            return "ممکن است اطلاعات ناقص باشد، اما بر اساس منابع موجود: " + candidate
                        return "متاسفانه اطلاعات کافی برای پاسخ به این سوال در پایگاه دانش موجود نیست."
                # Validate the provided final answer as well
                supported, reason = self.is_answer_supported(answer_text, chunks_cache or [])
                if supported:
                    return answer_text
                # Fallback: try to produce a validated response
                candidate = self.generate_response(question, chunks_cache or [], history)
                supported2, reason2 = self.is_answer_supported(candidate, chunks_cache or [])
                if supported2:
                    return candidate
                if chunks_cache and any([m.get('_low_confidence') for _, m in chunks_cache if isinstance(m, dict)]):
                    return "ممکن است اطلاعات ناقص باشد، اما بر اساس منابع موجود: " + candidate
                return "متاسفانه اطلاعات کافی برای پاسخ به این سوال در پایگاه دانش موجود نیست."

            # Unknown action; assume this is final text; validate
            supported, reason = self.is_answer_supported(response_text, chunks_cache or [])
            if supported:
                return response_text
            if chunks_cache and any([m.get('_low_confidence') for _, m in chunks_cache if isinstance(m, dict)]):
                return "ممکن است اطلاعات ناقص باشد، اما بر اساس منابع موجود: " + response_text
            return "متاسفانه اطلاعات کافی برای پاسخ به این سوال در پایگاه دانش موجود نیست."


    def generate_response(self, query: str, context_chunks: List[Tuple[str, Dict]], history: List[Dict]) -> str:
        """
        Generates a final response using the LLM based on the query and retrieved context (with metadata).
        """
        if not self.client:
            return "OpenAI client is not initialized. Cannot generate response."

        if not context_chunks:
            return "متاسفانه اطلاعات کافی برای پاسخ به سوال شما در پایگاه دانش من وجود ندارد."

        context_parts = []
        for i, (text, meta) in enumerate(context_chunks, 1):
            source_title = meta.get("source_title", "منبع نامشخص")
            parent_cat = meta.get("parent_category")
            ref = f"منبع {i}: بخش «{source_title}»"
            if parent_cat:
                ref += f" (زیرمجموعهٔ «{parent_cat}»)"
            url = meta.get("source_url", "").strip()
            if url and url != "N/A":
                ref += f" — لینک: {url}"
            context_parts.append(f"{ref}\nمتن مرتبط: {text}")

        context_string = "\n\n---\n\n".join(context_parts)

        # Build conversation history for LLM (last few turns)
        history_str = ""
        if history:
            history_str = "\n".join([
                f"{'کاربر' if turn['role'] == 'user' else 'پشتیبان'}: {turn['content']}"
                for turn in history[-4:]
            ])
            history_section = f"\n\nتاریخچه گفتگوی اخیر:\n{history_str}\n"
        else:
            history_section = ""

        system_prompt = """
        شما یک دستیار خدمات مشتری متخصص و دوستانه برای یک شرکت هستید. نام شما «پشتیبان» است.
        شما باید فقط و فقط بر اساس «زمینه» ارائه شده به «سوال» کاربر پاسخ دهید.
        اطلاعاتی را از خودتان اضافه نکنید. اگر زمینه برای پاسخ دادن کافی نیست،
        مودبانه بگویید که اطلاعات کافی برای پاسخ به آن سوال را ندارید.
        اگر در زمینه **آدرس سایت (URL)** ذکر شده باشد، می‌توانید بگویید:
        «برای اطلاعات بیشتر می‌توانید به بخش (عنوان بخش) در سایت ما مراجعه کنید: [لینک]»
        ***شما باید همیشه و فقط به زبان فارسی پاسخ دهید.***
        """

        user_prompt = f"زمینه: \"{context_string}\"\nتاریخچه گفتمان: {history_section}\n\nسوال فعلی: \"{query}\""

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
            )
            return completion.choices[0].message.content
        except:
            return "یک خطای غیرمنتظره رخ داد. لطفا دوباره تلاش کنید."