# backend/chatbot.py
import chromadb
from sentence_transformers import SentenceTransformer
import openai
import os
from typing import List, Tuple, Dict, Any, Optional
try:
    from chromadb.errors import NotFoundError
except ImportError:
    from chromadb.errors import InvalidCollectionException as NotFoundError

class RAGChatbot:
    def __init__(self, api_key: str, base_url: str = "https://api.tapsage.com/openai/v1"):
        # Embedding model
        self.embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        # Optional relevance gate; set to None to keep all results
        # Note: distance semantics depend on the Chroma collection metric.
        # We keep it configurable via env to avoid hardcoding a threshold.
        rt = os.environ.get("RELEVANCE_THRESHOLD")
        self.relevance_threshold = float(rt) if rt not in (None, "") else None

        # Whether to use the tool-calling agent loop. Default is OFF (safer, like the original code).
        self.agent_mode = os.environ.get("AGENT_MODE", "false").lower() in ("1", "true", "yes")
        
        # OpenAI client
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = "gpt-4o-mini"

        # ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        try:
            self.collection = self.chroma_client.get_collection(name="farsi_rag_collection")
        except (NotFoundError, ValueError) as exc:
            # Some chromadb versions raise ValueError when the collection is absent
            if isinstance(exc, ValueError) and "does not exist" not in str(exc):
                raise
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

    def is_greeting(self, text: str) -> bool:
        if not text:
            return False
        t = text.strip().lower()
        greetings = ["سلام", "hi", "hello", "درود", "hey"]
        return any(g in t for g in greetings)

    def _strip_tool_outputs(self, text: str) -> str:
        """Remove any tool-output markers from the final answer before sending to user."""
        if not text:
            return text
        markers = ["TOOL_OUTPUT_RETRIEVE", "TOOL_OUTPUT_SUMMARY"]
        lines = []
        for line in text.splitlines():
            if any(m in line for m in markers):
                continue
            lines.append(line)
        return "\n".join(lines).strip()

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

    def retrieve_relevant_chunks(self, queries: List[str], n_results: int = 6) -> List[Tuple[str, Dict[str, Any]]]:
        seen = set()
        results = []
        for q in queries:
            emb = self.embedding_model.encode(q).tolist()
            res = self.collection.query(query_embeddings=[emb], n_results=n_results, include=["documents", "metadatas", "distances"])
            docs = res.get('documents', [[]])[0]
            metas = res.get('metadatas', [[]])[0]
            dists = res.get('distances', [[]])[0] if res.get('distances') else [None] * len(docs)
            for doc, meta, dist in zip(docs, metas, dists):
                if self.relevance_threshold is not None and dist is not None and dist > self.relevance_threshold:
                    continue
                if doc not in seen:
                    seen.add(doc)
                    results.append((doc, meta))
        return results

    def answer(self, question: str, history: List[Dict[str, str]], allow_general_knowledge: bool = False) -> str:
        """Non-agent RAG path with strict retrieval gating (like the original version)."""
        standalone = self.rewrite_query(question, history)
        expanded = self.expand_query(standalone)
        chunks = self.retrieve_relevant_chunks(expanded)

        if not chunks:
            if self.is_greeting(question):
                return "سلام! خوشحال می‌شوم اگر دربارهٔ اطلاعات موجود سوال بپرسید."
            if allow_general_knowledge:
                return self.generate_response(question, [], history, allow_general_knowledge=True)
            return "متاسفانه اطلاعات کافی برای پاسخ به سوال شما در پایگاه دانش من وجود ندارد."

        return self.generate_response(question, chunks, history, allow_general_knowledge=allow_general_knowledge)

    def run_agent(
        self,
        question: str,
        history: List[Dict[str, str]],
        max_steps: int = 4,
        allow_general_knowledge: bool = False,
    ) -> str:
        """Answer a question.

        Default behavior (AGENT_MODE=false): strict retrieval gating + generate from retrieved chunks only.
        Optional behavior (AGENT_MODE=true): tool-calling loop.
        """

        if not self.agent_mode:
            return self.answer(question, history, allow_general_knowledge=allow_general_knowledge)

        """
        Agentic ReAct-style loop that lets the model call tools (retrieve/summarize) and then return a final answer.

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
                # If parsing fails, do NOT return a free-form answer in strict mode; fall back to gated RAG.
                return self.answer(question, history, allow_general_knowledge=allow_general_knowledge)

            if action == "retrieve":
                # Use the agent's retrieve tool
                if not payload:
                    messages.append({"role": "assistant", "content": "Action 'retrieve' requires 'input'."})
                    continue
                q_for_retrieve = payload
                res_chunks = self.retrieve_relevant_chunks([q_for_retrieve])
                chunks_cache = res_chunks
                # Append tool output to messages (keep concise and avoid leaking to user)
                output_text = "\n\n".join([f"[{i+1}] {t} (meta: {m})" for i, (t, m) in enumerate(res_chunks)])
                tool_outputs.append(output_text)
                messages.append({"role": "system", "content": f"TOOL_OUTPUT_RETRIEVE: {output_text}"})
                continue

            if action == "summarize":
                if chunks_cache is None:
                    messages.append({"role": "assistant", "content": "هیچ بخشی برای خلاصه کردن موجود نیست. ابتدا retrieve را اجرا کنید."})
                    continue
                summary = self.summarize_chunks(chunks_cache)
                tool_outputs.append(summary)
                messages.append({"role": "system", "content": f"TOOL_OUTPUT_SUMMARY: {summary}"})
                continue

            if action == "final":
                # If no relevant chunks and not a greeting, optionally allow general knowledge answers
                if (
                    not allow_general_knowledge
                    and (not chunks_cache or len(chunks_cache) == 0)
                    and not self.is_greeting(question)
                ):
                    return "این سوال در حوزهٔ داده‌های من نیست، لطفاً دربارهٔ مطالب موجود سوال کنید."
                # Generate the final answer from retrieved chunks to keep responses grounded.
                try:
                    return self.generate_response(
                        question,
                        chunks_cache or [],
                        history,
                        allow_general_knowledge=allow_general_knowledge,
                    )
                except Exception:
                    return "متاسفانه پاسخی تولید نشد."

            # Unknown action; return as final but sanitized
            return self._strip_tool_outputs(response_text)


    def generate_response(
        self,
        query: str,
        context_chunks: List[Tuple[str, Dict]],
        history: List[Dict],
        allow_general_knowledge: bool = False,
    ) -> str:
        """
        Generates a final response using the LLM based on the query and retrieved context (with metadata).
        """
        if not self.client:
            return "OpenAI client is not initialized. Cannot generate response."

        has_context = bool(context_chunks)

        if not has_context and not allow_general_knowledge:
            if self.is_greeting(query):
                return "سلام! خوشحال می‌شوم اگر دربارهٔ اطلاعات موجود سوال بپرسید."
            return "این سوال در حوزهٔ داده‌های من نیست، لطفاً دربارهٔ مطالب موجود سوال کنید."

        context_parts = []
        if has_context:
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
        else:
            # Allow LLM to answer cautiously without specific context when permitted
            context_parts.append("هیچ زمینهٔ مشخصی یافت نشد؛ در صورت نیاز با دانش عمومی و لحن محتاط پاسخ بده.")

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

        if allow_general_knowledge:
            system_prompt += "\nاگر زمینه‌ای موجود نیست، می‌توانید با دانش عمومی ولی با لحن محتاط و کوتاه پاسخ دهید."

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