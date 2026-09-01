"""Embedding 接口与不依赖第三方库的 Day 1 占位实现。"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence


class EmbeddingProvider(Protocol):
    """真实或占位 Embedding 模型必须遵守的接口。"""

    @property
    def model_name(self) -> str:
        """返回模型名称与版本，便于记录和复现。"""
        ...

    @property
    def dimension(self) -> int:
        """返回向量维度。"""
        ...

    def encode(self, text: str) -> list[float]:
        """把一段文本转换为定长向量。"""
        ...


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """计算两个等长向量的余弦相似度。"""

    if len(left) != len(right):
        raise ValueError("两个向量的维度必须相同")
    if not left:
        raise ValueError("向量不能为空")

    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)


class HashingEmbedder:
    """Day 1 的可运行占位 Embedding。

    它通过稳定哈希把中文字符和相邻字符映射到固定维度，目的是先验证
    Memory -> Embedding -> Retrieval 接口，不冒充真正的语义模型。
    Day 4 只需要用真实中文 Embedding 实现替换本类，无需修改检索主流程。
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("dimension 必须大于 0")
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return "day1-hashing-embedding-v1"

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, text: str) -> list[float]:
        normalized = re.sub(r"\s+", "", text.strip().lower())
        if not normalized:
            raise ValueError("待编码文本不能为空")

        vector = [0.0] * self._dimension
        features = list(normalized)
        features.extend(
            normalized[index : index + 2]
            for index in range(max(0, len(normalized) - 1))
        )

        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self._dimension
            sign = 1.0 if digest[0] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]
