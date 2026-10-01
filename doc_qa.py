"""私有文档知识库问答：读取本地文档 -> 分块 -> 检索 -> LLM 生成回答。

与项目 1 的区别：
1. 面向「整篇文档」而非「问答对」，需要先把文档切成语义块。
2. 检索返回最相关的文档片段，标注来源文件名。

零重依赖：文档解析仅支持 .txt 与 .md（UTF-8），检索用字符 bigram + TF 加权。
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------------
# 文档加载与分块
# ---------------------------------------------------------------------------

def load_documents(docs_dir: str) -> List[dict]:
    """加载目录下所有 .txt / .md 文档，返回 [{name, text}]。"""
    docs: List[dict] = []
    for ext in ("*.txt", "*.md"):
        for p in Path(docs_dir).glob(ext):
            docs.append({"name": p.name, "text": p.read_text(encoding="utf-8")})
    return docs


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """按字符数切块，带重叠，保证跨块语义不丢失。"""
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------

def _char_bigrams(text: str) -> set:
    text = text.lower()
    tokens: List[str] = []
    for chunk in re.findall(r"[\u4e00-\u9fff]+|[a-z]+|\d+", text):
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
            tokens.append(chunk)
        else:
            tokens.append(chunk)
    return set(tokens)


def search_chunks(query: str, chunks: List[dict], top_k: int = 3, threshold: float = 0.08) -> List[dict]:
    """在分块列表中检索，返回 [{text, source, score}]。chunks 元素含 name。"""
    q_grams = _char_bigrams(query)
    if not q_grams:
        return []
    scored: List[dict] = []
    for c in chunks:
        e_grams = _char_bigrams(c["text"])
        if not e_grams:
            continue
        overlap = len(q_grams & e_grams)
        union = q_grams | e_grams
        jaccard = overlap / len(union) if union else 0.0
        # 兜底：仅当 query 中存在「长度>=2」的 gram 完整出现在文档中才算字面命中，
        # 排除单字（如"天"）造成的误召回。
        containment = any(len(g) >= 2 and g in c["text"].lower() for g in q_grams)
        score = max(jaccard, 0.15 if containment else 0.0)
        scored.append({"text": c["text"], "source": c["name"], "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return [s for s in scored[:top_k] if s["score"] >= threshold]


# ---------------------------------------------------------------------------
# LLM 调用（与项目 1 相同的可配置客户端）
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self, api_key=None, api_base=None, model=None):
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.api_base = (api_base or os.environ.get("LLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4")).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "glm-4-flash")

    def chat(self, messages, temperature=0.3):
        if not self.api_key:
            raise RuntimeError("未配置 LLM_API_KEY，请设置环境变量后重试。")
        url = f"{self.api_base}/chat/completions"
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# QA 引擎
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "你是一个文档问答助手。请仅根据提供的文档片段回答用户问题，"
    "回答要准确、简洁，并在结尾用「（来源：文件名）」标注依据。"
    "如果文档片段中找不到答案，请明确说明「文档中没有相关信息」，不要编造。"
)


class DocQA:
    def __init__(self, docs_dir: str, llm: LLMClient | None = None, chunk_size=300, overlap=50):
        self.docs_dir = docs_dir
        self.llm = llm or LLMClient()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[dict] = self._build_index()

    def _build_index(self) -> List[dict]:
        chunks: List[dict] = []
        for doc in load_documents(self.docs_dir):
            for piece in chunk_text(doc["text"], self.chunk_size, self.overlap):
                chunks.append({"text": piece, "name": doc["name"]})
        return chunks

    def ask(self, question: str) -> dict:
        hits = search_chunks(question, self.chunks)
        if not hits:
            return {"question": question, "answer": "文档中没有相关信息。", "sources": []}
        context = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"文档片段：\n{context}\n\n问题：{question}"},
        ]
        answer = self.llm.chat(messages)
        return {"question": question, "answer": answer, "sources": [(h["source"], h["score"]) for h in hits]}
