import os
import time
import logging
import requests


logger = logging.getLogger(__name__)


class Qwen3:
    """轻量客户端：直接以 OpenAI 兼容的 HTTP 接口调用 /v1/chat/completions。

    环境变量（可选）：
    - QWEN_BASE_URL: 例如 http://10.210.1.23:9014/v1
    - QWEN_MODEL:    例如 qwen3-32b
    - OPENAI_API_KEY: 用于 Authorization: Bearer <key>；若不需要鉴权，保持默认 'empty'
    - HTTP(S)_PROXY: 若需要代理，可使用 requests 的环境变量方式
    """

    def __init__(self,
                 base_url: str | None = None,
                 model: str | None = None,
                 api_key: str | None = None,
                 timeout: float = 60.0,
                 max_retries: int = 2):
        self.base_url = (base_url or os.getenv('QWEN_BASE_URL') or 'http://10.210.1.23:10000/v1').rstrip('/')
        self.model = model or os.getenv('QWEN_MODEL') or 'Qwen3-14B'
        self.api_key = api_key or os.getenv('OPENAI_API_KEY') or 'empty'
        # 允许通过环境变量覆盖默认超时与重试次数
        try:
            env_timeout = float(os.getenv('QWEN_TIMEOUT_SEC', '').strip())
            if env_timeout > 0:
                timeout = env_timeout
        except Exception:
            pass
        try:
            env_retries = int(os.getenv('QWEN_MAX_RETRIES', '').strip())
            if env_retries >= 0:
                max_retries = env_retries
        except Exception:
            pass
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def __call__(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {self.api_key}",
        }
        # enable_think = os.getenv('QWEN_ENABLE_THINK', '').lower() in ('1', 'true', 'yes', 'on')
        payload = {
            'model': self.model,
            'messages': [{
                'role': 'user',
                'content': prompt,
            }],
            'max_tokens': 32768,
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 20,
            'min_p': 0.0,
            'presence_penalty': 1.5,
            'chat_template_kwargs': {'enable_thinking': False},  # 直接顶层传递
        }
        # # 可按需启用思维链（兼容部分实现的 extra_body 透传）
        # if enable_think:
        #     payload['extra_body'] = {'chat_template_kwargs': {'enable_thinking': True}}
        # else:
        #     payload['extra_body'] = {'chat_template_kwargs': {'enable_thinking': False}}
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                # 兼容 OpenAI 格式
                content = (
                    data.get('choices', [{}])[0]
                        .get('message', {})
                        .get('content')
                )
                if not content:
                    # 某些实现将文本放在 'text'
                    content = data.get('choices', [{}])[0].get('text')
                if not isinstance(content, str):
                    raise ValueError(f"unexpected response format: {data}")
                return content
            except Exception as e:
                last_err = e
                # 轻量退避
                time.sleep(0.5 * (attempt + 1))
        # 若到此仍失败，抛出最后一次的异常
        raise RuntimeError(f"qwen chat.completions failed after retries: {last_err}")


if __name__ == '__main__':
    llm = Qwen3()
    prompt = "什么是 LLM？请用一句话回答。"
    try:
        print(llm(prompt))
    except Exception as e:
        print(f"request failed: {e}")