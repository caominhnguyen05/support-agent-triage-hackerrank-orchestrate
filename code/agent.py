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

load_dotenv()

# Config
BASE_DIR = Path(__file__).parent.parent / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "support_corpus"

TOP_K = 6
SIMILARITY_THRESHOLD = 0.4

# Topics that should always be escalated to a human agent
ESCALATION_KEYWORDS = [
    "fraud", "unauthorized", "stolen", "hacked", "compromised",
    "chargeback", "dispute", "refund", "lawsuit", "legal action",
    "legal notice", "court", "police", "death", "deceased",
    "cannot log in", "locked out", "account suspended", "banned",
    "discrimination", "harassment", "abuse", "threat",
]

DOMAIN_HINTS = {
    "hackerrank": ["hackerrank", "coding test", "assessment", "challenge",
                   "interview", "hiring", "recruiter", "test link", "proctoring"],
    "claude":     ["claude", "anthropic", "ai assistant", "subscription",
                   "pro plan", "claude.ai", "artifact", "conversation"],
    "visa":       ["visa", "card", "payment", "transaction", "merchant",
                   "chargeback", "cvv", "pin", "atm", "international"],
}


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

    # normalize encoding issues (optional safety)
    text = text.encode("utf-8", errors="ignore").decode("utf-8")

    return text

class TriageAgent:

    def __init__(self):
        self.openai_client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]
        )
        self.embedder = get_embedder()

        # Connect to ChromaDB
        self.chroma = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma.get_collection(COLLECTION_NAME)
        print(f"[agent] Loaded collection with {self.collection.count()} chunks")

    def _classify_domain(self, issue: str, subject: str, company: str) -> str:
        """
        Return one of: hackerrank | claude | visa | general
        If company is a known value, use it directly.
        Otherwise, count hint-word matches across domains.
        """
        company_lower = (company or "").strip().lower()

        if company_lower == "hackerrank":
            return "hackerrank"
        if company_lower == "claude":
            return "claude"
        if company_lower == "visa":
            return "visa"

        combined = f"{issue} {subject}".lower()
        scores = {domain: 0 for domain in DOMAIN_HINTS}
        for domain, hints in DOMAIN_HINTS.items():
            for hint in hints:
                if hint in combined:
                    scores[domain] += 1

        best_domain = max(scores, key=lambda d: scores[d])
        return best_domain if scores[best_domain] > 0 else "general"

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
    def _should_escalate(self, issue: str, subject: str, chunks: list[dict]) -> tuple[bool, str]:
        """
        Return (should_escalate: bool, reason: str).
        Escalation triggers:
          1. Known high-risk keywords in the ticket
          2. Max retrieval similarity below threshold (corpus can't answer it)
        """
        combined = f"{issue} {subject}".lower()

        for kw in ESCALATION_KEYWORDS:
            if kw in combined:
                return True, f"Sensitive topic detected: '{kw}'"

        if not chunks:
            return True, "No relevant documentation found in corpus"

        max_sim = max(c["similarity"] for c in chunks)
        if max_sim < SIMILARITY_THRESHOLD:
            return (
                True,
                f"Low retrieval confidence (max similarity {max_sim:.2f} < threshold {SIMILARITY_THRESHOLD})"
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
    ) -> dict:
        """
        Call OpenAI's gpt-4o-mini to generate a JSON-structured triage decision.
        The system prompt enforces grounded answers only.
        """

        context_block = "\n\n".join(
            f"[Source: {c['title']} | {c['source']}]\n{c['text']}"
            for c in chunks
        )

        system_prompt = """You are a multi-domain support agent.
        Your job is to analyse support tickets and produce a structured JSON response.

        STRICT RULES:
        1. Base your response ONLY on the documentation excerpts provided in <context>.
        2. Never invent policies, prices, procedures, or capabilities not in the context.
        3. If context is insufficient or you cannot cite context, you MUST escalate to a human agent.
        4. Be professional, concise, and empathetic.
        5. Every claim in the response MUST be traceable to the context.

        Respond with a single JSON object with these fields:
        {
        "status":       "replied" | "escalated",
        "product_area": "<most relevant support category, e.g. Billing, Account Access, API, Assessments, Card Services>",
        "response":     "<user-facing answer OR escalation message>",
        "justification": "<2-3 sentences explaining your decision and you must cite sources used>",
        "request_type": "product_issue" | "feature_request" | "bug" | "invalid"
        }

        For status:
        - Use "escalated" when the issue is sensitive, requires human verification,
            involves account security, fraud, legal matters, or is outside the corpus.
        - Use "replied" when the corpus clearly answers the question.

        For request_type:
        - product_issue: user is asking about how something works or having trouble
        - feature_request: user is asking for a new capability
        - bug: user reports something is broken / unexpected behaviour
        - invalid: ticket is spam, gibberish, test, or clearly outside scope

        RESPONSE FORMAT RULES (IMPORTANT):
        - "response" MUST be multi-line plain text.
        - Each paragraph MUST be separated by a blank line (\n\n).
        - If steps exist, each step MUST be on its own line and start with a number (1. 2. 3.).
        - Do NOT compress multiple steps into one line.
        - Do NOT use markdown of any kind.
        - Do NOT use bullet symbols or special typography.
        - Use only plain text characters.
        - Keep response structured like a help article (paragraphs + line breaks only).
        """

        user_prompt = f"""Support ticket details:
        Subject: {subject or '(none)'}
        Company: {domain}
        Issue: {issue}

        {'NOTE: This ticket has been flagged for potential escalation. Reason: ' + escalate_reason if escalate else ''}

        <context>
        {context_block if context_block else 'No relevant documentation found.'}
        </context>

        Instructions:
        - Decide reply or escalate
        - If replying, write structured multi-line response:
        paragraph spacing required (blank lines between paragraphs)
        steps must be line-separated (not compressed)
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
            # Fallback: extract what we can
            result = {
                "status":        "escalated",
                "product_area":  domain.title() + " Support",
                "response":      "We were unable to process this request automatically. A human agent will follow up.",
                "justification": f"JSON parsing failed. Raw model output: {raw[:300]}",
                "request_type":  "product_issue",
            }

        # Enforce escalation if the safety gate triggered
        if escalate and result.get("status") != "escalated":
            result["status"] = "escalated"
            result["justification"] = (
                escalate_reason + " | " + result.get("justification", "")
            )

        return result


    def triage(self, issue: str, subject: str, company: str) -> dict:
        """
        Full five-step pipeline. Returns a dict with:
          status, product_area, response, justification, request_type
        """
        # Step 1: domain
        domain = self._classify_domain(issue, subject, company)

        # Step 2: retrieve
        query  = f"{subject or ''} {issue}".strip()
        chunks = self._retrieve(query, domain)

        # Step 3: escalation check
        escalate, escalate_reason = self._should_escalate(issue, subject, chunks)

        # Steps 4+5: generate structured response
        result = self._generate_response(
            issue=issue,
            subject=subject,
            domain=domain,
            chunks=chunks,
            escalate=escalate,
            escalate_reason=escalate_reason,
        )

        return result