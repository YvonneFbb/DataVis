"""Doubao vision client for post-OCR filtering."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import base64
import os
import time

from openai import OpenAI


@dataclass
class ArkVisionConfig:
    base_url: str
    model: str
    api_key_env: str = 'ARK_API_KEY'
    timeout: int = 60
    temperature: float = 0.0
    max_tokens: int = 256
    system_prompt: str = (
        "你是古籍字符质量的终审员。请非常严格地审查输入图像，遵守以下规则：\n"
        "1. 只能依据图像中真实可见的笔画判断，不允许凭经验或模糊轮廓猜测。\n"
        "2. 若字符缺笔、断笔、模糊、被截断、遮挡，或无法完全辨认，应判定 `recognizable=false`，并令 `character` 为空字符串。\n"
        "3. 框选必须紧贴单个汉字。图像四周不允许出现任何多余内容：空白边框、其他字符/笔画、装饰、水印或噪点，一旦存在就判定 `has_noise=true`，并视情况令 `single_character=false` 或 `recognizable=false`。\n"
        "4. 输入图像预期是单个汉字。字符识别结果必须是一个汉字；若不是汉字或不确定，应返回空字符串并视为不合格。\n"
        "5. `comment` 用于说明问题原因或额外备注。\n"
        "请仅输出一个 JSON，字段：single_character(bool), recognizable(bool), has_noise(bool), character(string), comment(string)。"
    )


class ArkVisionClient:
    def __init__(self, cfg: ArkVisionConfig):
        api_key = os.getenv(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(f"环境变量 {cfg.api_key_env} 未设置，无法调用 Ark 模型")
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=api_key)

    def _encode_image(self, path: str) -> str:
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def analyse_character(self, image_path: str) -> Dict[str, Any]:
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": self.cfg.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self._encode_image(image_path)}"}},
                        {"type": "text", "text": "只输出上述 JSON"},
                    ],
                },
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        start = time.time()
        response = self.client.chat.completions.create(**payload)
        duration = time.time() - start
        content = response.choices[0].message.content.strip()
        return {
            "raw": content,
            "response_time": duration,
        }
