"""
AI Agent 工具调用助手
支持：时间查询、数学计算、翻译、天气查询、向量检索
"""

import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

# 加载环境变量
load_dotenv()


# ============ 阿里云 Embedding 客户端 ============
class DashScopeEmbeddings:
    """阿里云 DashScope Embedding 客户端"""

    def __init__(self, model: str = "text-embedding-v2"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    def embed_query(self, text: str) -> list:
        """获取单个文本的 embedding"""
        # 使用 OpenAI 兼容接口
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "input": text
        }
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def embed_documents(self, texts: list) -> list:
        """获取多个文本的 embedding"""
        return [self.embed_query(text) for text in texts]


# ============ 向量检索模块 ============
class VectorStore:
    """简单的向量存储（内存版）"""

    def __init__(self, embedding_model: str = "text-embedding-v2"):
        self.embeddings = DashScopeEmbeddings(model=embedding_model)
        self.documents = []  # 存储文档
        self.vectors = []    # 存储向量

    def add_document(self, text: str, metadata: dict = None):
        """添加文档"""
        vector = self.embeddings.embed_query(text)
        self.documents.append({"text": text, "metadata": metadata or {}})
        self.vectors.append(vector)

    def add_documents(self, texts: list, metadatas: list = None):
        """批量添加文档"""
        for i, text in enumerate(texts):
            self.add_document(text, metadatas and metadatas[i])

    def similarity_search(self, query: str, k: int = 3) -> list:
        """相似度搜索"""
        if not self.documents:
            return []

        query_vector = self.embeddings.embed_query(query)

        # 计算余弦相似度
        similarities = []
        for i, doc_vector in enumerate(self.vectors):
            similarity = self._cosine_similarity(query_vector, doc_vector)
            similarities.append((similarity, self.documents[i]))

        # 按相似度排序，返回前k个
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [doc for sim, doc in similarities[:k]]

    def _cosine_similarity(self, v1: list, v2: list) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a ** 2 for a in v1) ** 0.5
        norm2 = sum(b ** 2 for b in v2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 and norm2 else 0


# 向量存储实例（延迟初始化）
_vector_store = None

def get_vector_store():
    """获取向量存储实例"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


# ============ 工具定义 ============
@tool
def get_current_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """数学计算，输入数学表达式如 '2+3*4'"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"


@tool
def translate(text: str, target_lang: str = "zh") -> str:
    """翻译文本到目标语言"""
    return f"翻译结果({target_lang})：{text}"


@tool
def get_weather(city: str = "北京") -> str:
    """查询城市天气"""
    return f"{city}：晴天，25℃"


@tool
def search_knowledge(query: str) -> str:
    """从知识库中检索相关信息"""
    results = get_vector_store().similarity_search(query, k=3)
    if not results:
        return "知识库中没有找到相关信息"

    output = "检索到以下相关内容：\n"
    for i, doc in enumerate(results, 1):
        output += f"{i}. {doc['text']}\n"
    return output


@tool
def add_to_knowledge(text: str) -> str:
    """将文本添加到知识库"""
    get_vector_store().add_document(text)
    return f"已添加到知识库：{text[:50]}..."


# ============ Agent 类封装 ============
class ToolAgent:
    """AI 工具调用助手，支持向量检索"""

    def __init__(self, model_name: str = "qwen-max", temperature: float = 0):
        """
        初始化 Agent

        Args:
            model_name: 模型名称
            temperature: 温度参数
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.tools = [
            get_current_time,
            calculate,
            translate,
            get_weather,
            search_knowledge,
            add_to_knowledge
        ]
        self.agent = create_agent(self.llm, self.tools)
        self.conversation_history = []

    def add_document(self, text: str):
        """直接添加文档到知识库"""
        get_vector_store().add_document(text)

    def add_documents(self, texts: list):
        """批量添加文档"""
        get_vector_store().add_documents(texts)

    def chat(self, user_input: str) -> str:
        """与 Agent 对话"""
        messages = self.conversation_history + [{"role": "user", "content": user_input}]
        result = self.agent.invoke({"messages": messages})
        last_message = result["messages"][-1]

        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": last_message.content})

        return last_message.content

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []

    def run(self):
        """启动交互式对话"""
        print("=== AI 工具调用助手 ===")
        print("可用工具：时间、计算、翻译、天气、向量检索")
        print("命令：exit退出 | clear清空 | add添加知识")
        print("=" * 30)

        while True:
            user_input = input("\n你：").strip()

            if not user_input:
                continue

            if user_input.lower() == "exit":
                print("再见！")
                break

            if user_input.lower() == "clear":
                self.clear_history()
                print("对话历史已清空")
                continue

            # 直接添加知识
            if user_input.startswith("add "):
                text = user_input[4:]
                self.add_document(text)
                print(f"已添加：{text}")
                continue

            try:
                response = self.chat(user_input)
                print(f"\nAI：{response}")
            except Exception as e:
                print(f"\n错误：{e}")


# ============ 使用示例 ============
if __name__ == "__main__":
    agent = ToolAgent(model_name="qwen-max")

    # 预添加一些知识
    agent.add_documents([
        "南阳师范学院位于河南省南阳市，是一所公办本科院校",
        "软件工程专业主要学习Python、Java等编程语言",
        "LangChain是构建AI Agent的流行框架",
    ])

    agent.run()