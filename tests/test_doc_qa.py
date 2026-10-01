"""文档问答单元测试（不依赖真实 LLM）。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from doc_qa import DocQA, LLMClient, chunk_text, load_documents, search_chunks  # noqa: E402


class FakeLLM(LLMClient):
    def chat(self, messages, temperature=0.3):
        return f"[mock] 回答: {messages[-1]['content'][:40]}"


class TestChunking(unittest.TestCase):
    def test_chunk_text(self):
        chunks = chunk_text("a" * 700, chunk_size=300, overlap=50)
        self.assertGreaterEqual(len(chunks), 3)

    def test_empty(self):
        self.assertEqual(chunk_text(""), [])


class TestLoad(unittest.TestCase):
    def test_load_documents(self):
        docs_dir = str(Path(__file__).parent.parent / "docs")
        docs = load_documents(docs_dir)
        self.assertGreaterEqual(len(docs), 1)
        self.assertTrue(all("name" in d and "text" in d for d in docs))


class TestDocQA(unittest.TestCase):
    def setUp(self):
        docs_dir = str(Path(__file__).parent.parent / "docs")
        self.qa = DocQA(docs_dir, FakeLLM())

    def test_ask_hit(self):
        r = self.qa.ask("续航多久")
        self.assertIn("mock", r["answer"])
        self.assertTrue(len(r["sources"]) >= 1)

    def test_ask_miss(self):
        # 用与文档无任何字面重叠的查询，验证真正 miss 时返回兜底
        r = self.qa.ask("股票今天涨停了吗")
        self.assertIn("没有相关信息", r["answer"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
