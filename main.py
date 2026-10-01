"""命令行入口：文档问答交互。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from doc_qa import DocQA, LLMClient  # noqa: E402


def main():
    docs_dir = str(Path(__file__).parent / "docs")
    engine = DocQA(docs_dir, LLMClient())
    print("=" * 50)
    print("文档知识库问答已启动")
    print("已加载", len(engine.chunks), "个文档片段")
    print("输入问题开始提问，输入 exit / quit 退出")
    print("=" * 50)
    while True:
        try:
            q = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "退出"):
            print("再见！")
            break
        try:
            r = engine.ask(q)
            print(f"\n助手：{r['answer']}")
        except RuntimeError as e:
            print(f"\n[错误] {e}")
            break


if __name__ == "__main__":
    main()
