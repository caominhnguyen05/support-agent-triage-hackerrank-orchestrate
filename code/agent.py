"""
agent.py

  The core triage pipeline logic. Given a single support ticket (issue, subject,
  company), it runs a five-step pipeline and returns a structured result:
    1. Domain classification
    2. Retrieval of relevant documentation chunks from ChromaDB
    3. Decide to answer or escalate based on safety checks and retrieval confidence
    4. Generate a structured response using OpenAI's gpt-4o-mini
    5. Parse and return the final output fields
"""

import json
import os
import re
from pathlib import Path
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI
from rules import HARD_ESCALATION_KEYWORDS, SOFT_ESCALATION_KEYWORDS, DOMAIN_HINTS
from classifier import DomainClassifier, classify_product_area, classify_request_type, _word_match

load_dotenv()

# Config
BASE_DIR = Path(__file__).parent.parent / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "support_corpus"

TOP_K = 6
SIMILARITY_THRESHOLD = 0.35

def get_embedder():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def embed(texts):
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in resp.data]

    return embed

def clean_llm_output(text: str) -> str:
    # remove markdown bold
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

    # remove stray markdown symbols
    text = text.replace("*", "")
    text = text.replace("###", "")
    text = text.replace("##", "")

    # normalize encoding issues
    text = text.encode("utf-8", errors="ignore").decode("utf-8")

    return text

class TriageAgent:

    def __init__(self):
        self.openai_client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]
        )
        self.embedder = get_embedder()
        self.classifier = DomainClassifier()

        self.chroma = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma.get_collection(COLLECTION_NAME)
        print(f"[agent] Loaded collection with {self.collection.count()} chunks")

    def _classify_domain(self, issue: str, subject: str, company: str) -> str:
        """
        Three-stage classification:
          1. Exact company field match
          2. TF-IDF classifier on issue + subject text
          3. Keyword hint counting (fallback when classifier is uncertain)
        """
        company_lower = (company or "").strip().lower()

        if company_lower in ("hackerrank", "claude", "visa"):
            return company_lower

        # ML classifier
        combined = f"{subject or ''} {issue}".strip()
        domain, confidence = self.classifier.predict(combined)
        if domain != "uncertain":
            print(f"[classifier] domain={domain} confidence={confidence:.2f}")
            return domain

        print(f"[classifier] low confidence ({confidence:.2f}), falling back to keywords")
        combined_lower = combined.lower()
        scores = {d: 0 for d in DOMAIN_HINTS}
        for d, hints in DOMAIN_HINTS.items():
            for hint in hints:
                if _word_match(hint.lower(), combined_lower):
                    scores[d] += 1

        best = max(scores, key=lambda d: scores[d])
        return best if scores[best] > 0 else "general"

    # Finds the most relevant documentation chunks
    def _retrieve(self, query: str, domain: str) -> list[dict]:
        """
        Embed the query, search ChromaDB, and return up to TOP_K results.
        Tries domain-filtered search first, falls back to global search.
        """
        query_embedding = self.embedder([query])[0]

        where_filter = {"domain": {"$eq": domain}} if domain != "general" else None

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=TOP_K,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # Fallback: no domain filter
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=TOP_K,
                include=["documents", "metadatas", "distances"],
            )

        chunks = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                similarity = 1.0 - dist
                chunks.append({
                    "text": doc,
                    "source": meta.get("source", ""),
                    "title": meta.get("title", ""),
                    "domain": meta.get("domain", ""),
                    "similarity": round(similarity, 4),
                })

        return chunks

    # Decide if the agent should answer automatically or escalate to a human
    def _should_escalate(
        self, issue: str, subject: str, chunks: list[dict], domain: str
    ) -> tuple[bool, str]:
        combined = f"{issue} {subject}".lower()

        # Hard escalation — no corpus can handle these safely
        for kw in HARD_ESCALATION_KEYWORDS:
            if kw in combined:
                return True, f"Hard escalation: sensitive topic '{kw}'"

        has_soft_risk = any(kw in combined for kw in SOFT_ESCALATION_KEYWORDS)

        # No chunks found -> always escalate
        if not chunks:
            return True, "No relevant documentation found in corpus"

        max_sim = max(c["similarity"] for c in chunks)
        threshold = SIMILARITY_THRESHOLD

        # Soft risk + low retrieval -> escalate
        if has_soft_risk and max_sim < threshold:
            return (
                True,
                f"Soft-risk topic with low retrieval confidence "
                f"(max similarity {max_sim:.2f} < {threshold})",
            )

        # No risk keywords, but retrieval too weak → escalate
        if not has_soft_risk and max_sim < threshold:
            return (
                True,
                f"Low retrieval confidence "
                f"(max similarity {max_sim:.2f} < threshold {threshold})",
            )

        return False, ""

    def _generate_response(
        self,
        issue: str,
        subject: str,
        domain: str,
        chunks: list[dict],
        escalate: bool,
        escalate_reason: str,
        pre_product_area: str | None = None, 
        pre_request_type: str | None = None,
    ) -> dict:

        context_block = "\n\n".join(
            f"[Source: {c['title']} | {c['source']}]\n{c['text']}"
            for c in chunks
        )

        # Tell the LLM which fields are already decided so it doesn't contradict them
        pre_classified_note = ""
        if pre_product_area:
            pre_classified_note += f'\n IMPORTANT: product_area is already determined: "{pre_product_area}". Use this value exactly.'
        if pre_request_type:
            pre_classified_note += f'\n IMPORTANT: request_type is already determined: "{pre_request_type}". Use this value exactly.'

        system_prompt = f"""You are a multi-domain support agent.
            Your job is to analyse support tickets and produce a structured JSON response.

            STRICT RULES:
            1. Base your response ONLY on the documentation excerpts provided in <context>.
            2. Never invent policies, prices, procedures, or capabilities not in the context.
            3. If context is insufficient or you cannot cite context, you MUST escalate to a human agent.
            4. Be professional, concise, and empathetic.
            5. Every claim in the response MUST be traceable to the context.
            {pre_classified_note}

            Respond with a single JSON object with these fields:
            {{
            "status":       "replied" | "escalated",
            "product_area": "<use the pre-determined value if given, otherwise most relevant category from the context>",
            "response":     "<user-facing answer OR escalation message>",
            "justification": "<2-3 sentences explaining your decision and you must cite sources used>",
            "request_type": "<use the pre-determined value if given, otherwise: product_issue | feature_request | bug | invalid>"
            }}

            For status:
            - Use "escalated" ONLY when the issue requires human verification, involves account security,
                fraud, legal matters, or the corpus clearly cannot answer it.
            - Use "replied" when the corpus answers the question, even if the topic involves sensitive
                keywords (e.g. a stolen card can be replied to if the corpus provides contact instructions).

            RESPONSE FORMAT RULES:
            - "response" must be multi-line plain text.
            - Each paragraph MUST be separated by a blank line (\\n\\n).
            - If steps exist, each step MUST be on its own line and start with a number (1. 2. 3.).
            - Do NOT use markdown of any kind.
            - Use only plain text characters.
            """

        user_prompt = f"""Support ticket details:
            Subject: {subject or '(none)'}
            Company: {domain}
            Issue: {issue}

            {'NOTE: This ticket has been flagged for escalation. Reason: ' + escalate_reason if escalate else ''}

            <context>
            {context_block if context_block else 'No relevant documentation found.'}
            </context>

            Instructions:
            - Decide reply or escalate
            - If replying, write structured multi-line response
            - Output ONLY JSON"""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=1000,
        )

        raw = response.choices[0].message.content.strip()
        raw = clean_llm_output(raw)

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {
                "status":        "escalated",
                "product_area":  domain.title() + " Support",
                "response":      "We were unable to process this request automatically. A human agent will follow up.",
                "justification": f"JSON parsing failed. Raw model output: {raw[:300]}",
                "request_type":  "product_issue",
            }

        # Make sure LLM doesn't override the pre-classified fields
        if pre_product_area:
            result["product_area"] = pre_product_area
        if pre_request_type:
            result["request_type"] = pre_request_type

        # Enforce escalation if the safety gate triggered
        if escalate and result.get("status") != "escalated":
            result["status"] = "escalated"
            result["justification"] = escalate_reason + " | " + result.get("justification", "")

        return result


    def triage(self, issue: str, subject: str, company: str) -> dict:
        # Step 1: domain
        domain = self._classify_domain(issue, subject, company)

        # Step 2: retrieve
        query  = f"{subject or ''} {issue}".strip()
        chunks = self._retrieve(query, domain)

        # Step 2b: pre-classify product_area and request_type before LLM
        pre_product_area = classify_product_area(domain, issue, subject)
        pre_request_type = classify_request_type(issue, subject)

        if pre_product_area:
            print(f"[classify] product_area={pre_product_area!r}")
        if pre_request_type:
            print(f"[classify] request_type={pre_request_type!r}")

        # Step 3: escalation check
        escalate, escalate_reason = self._should_escalate(issue, subject, chunks, domain)

        # Steps 4+5: generate structured response
        result = self._generate_response(
            issue=issue,
            subject=subject,
            domain=domain,
            chunks=chunks,
            escalate=escalate,
            escalate_reason=escalate_reason,
            pre_product_area=pre_product_area,
            pre_request_type=pre_request_type,
        )

        return result