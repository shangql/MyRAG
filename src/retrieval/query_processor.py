"""查询预处理模块 - 中文分词、同义词扩展、查询改写"""
from typing import Dict, List, Optional, Set
import jieba
import jieba.analyse
from dataclasses import dataclass, field

from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryProcessResult:
    """查询处理结果

    Attributes:
        original: 原始查询
        processed: 处理后的查询
        tokens: 分词结果
        keywords: 关键词列表
        expanded: 扩展后的查询
    """
    original: str
    processed: str
    tokens: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    expanded: str = ""


class QueryProcessor:
    """查询处理器

    负责对用户查询进行预处理，提升检索效果。
    支持中文分词、关键词提取、同义词扩展。

    Attributes:
        use_stopwords: 是否使用停用词过滤
        stopwords: 停用词集合
        synonym_dict: 同义词词典
    """

    # 默认停用词列表
    DEFAULT_STOPWORDS: Set[str] = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
        "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
        "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "么",
    }

    def __init__(
        self,
        use_stopwords: bool = True,
        custom_stopwords: Optional[Set[str]] = None,
    ):
        """初始化查询处理器

        Args:
            use_stopwords: 是否使用停用词过滤
            custom_stopwords: 自定义停用词集合
        """
        self.use_stopwords = use_stopwords
        self.stopwords = self.DEFAULT_STOPWORDS.copy()
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)

        self.synonym_dict: Dict[str, List[str]] = {}
        self._init_jieba()

    def _init_jieba(self) -> None:
        """初始化 jieba 分词器"""
        # 设置日志级别，减少输出
        jieba.setLogLevel(jieba.logging.INFO)
        logger.debug("Jieba 分词器初始化完成")

    def process(self, query: str) -> QueryProcessResult:
        """处理查询

        完整的查询处理流程：
        1. 分词
        2. 停用词过滤
        3. 关键词提取
        4. 同义词扩展

        Args:
            query: 原始查询文本

        Returns:
            QueryProcessResult: 处理结果
        """
        if not query or not query.strip():
            return QueryProcessResult(
                original=query or "",
                processed="",
                tokens=[],
                keywords=[],
                expanded="",
            )

        # 1. 分词
        tokens = self.tokenize(query)

        # 2. 过滤停用词
        if self.use_stopwords:
            tokens = self._filter_stopwords(tokens)

        # 3. 关键词提取
        keywords = self.extract_keywords(query, topk=5)

        # 4. 同义词扩展
        expanded_tokens = self._expand_synonyms(tokens)
        expanded_query = " ".join(expanded_tokens)

        # 5. 构建处理后的查询
        processed = " ".join(tokens)

        return QueryProcessResult(
            original=query,
            processed=processed,
            tokens=tokens,
            keywords=keywords,
            expanded=expanded_query,
        )

    def tokenize(self, query: str) -> List[str]:
        """中文分词

        Args:
            query: 查询文本

        Returns:
            分词后的词列表
        """
        return list(jieba.cut(query))

    def _filter_stopwords(self, tokens: List[str]) -> List[str]:
        """过滤停用词

        Args:
            tokens: 分词结果

        Returns:
            过滤后的词列表
        """
        return [t for t in tokens if t.strip() and t not in self.stopwords]

    def extract_keywords(
        self,
        query: str,
        topk: int = 5,
        method: str = "textrank",
    ) -> List[str]:
        """提取关键词

        使用 TextRank 或 TF-IDF 算法提取关键词。

        Args:
            query: 查询文本
            topk: 返回关键词数量
            method: 提取方法 (textrank/tfidf)

        Returns:
            关键词列表
        """
        if method == "tfidf":
            keywords = jieba.analyse.extract_tags(
                query,
                topK=topk,
                withWeight=False,
            )
        else:
            keywords = jieba.analyse.textrank(
                query,
                topK=topk,
                withWeight=False,
            )

        return keywords

    def _expand_synonyms(self, tokens: List[str]) -> List[str]:
        """同义词扩展

        Args:
            tokens: 分词结果

        Returns:
            扩展后的词列表
        """
        expanded = list(tokens)

        for token in tokens:
            if token in self.synonym_dict:
                expanded.extend(self.synonym_dict[token])

        # 去重但保持顺序
        seen = set()
        result = []
        for word in expanded:
            if word not in seen:
                seen.add(word)
                result.append(word)

        return result

    def add_synonym(self, word: str, synonyms: List[str]) -> None:
        """添加同义词

        Args:
            word: 词语
            synonyms: 同义词列表
        """
        self.synonym_dict[word] = synonyms

    def load_synonym_dict(self, synonyms: Dict[str, List[str]]) -> None:
        """批量加载同义词词典

        Args:
            synonyms: 同义词字典 {word: [synonyms]}
        """
        self.synonym_dict.update(synonyms)
        logger.info(f"加载同义词词典，包含 {len(synonyms)} 个词条")

    def get_query_vector_terms(self, query: str) -> List[str]:
        """获取查询向量词项

        用于混合检索中的关键词检索。

        Args:
            query: 查询文本

        Returns:
            词项列表
        """
        result = self.process(query)
        # 合并分词结果和关键词
        terms = list(set(result.tokens + result.keywords))
        return terms


# 全局查询处理器实例
_query_processor: Optional[QueryProcessor] = None


def get_query_processor() -> QueryProcessor:
    """获取全局查询处理器实例（单例）

    Returns:
        QueryProcessor: 查询处理器实例
    """
    global _query_processor
    if _query_processor is None:
        _query_processor = QueryProcessor()
    return _query_processor
