import logging
import json
from typing import AsyncGenerator, Optional

from openai import AsyncOpenAI, APIError
from anthropic import AsyncAnthropic, AnthropicError

from src.config.settings import settings

logger = logging.getLogger("hyperrag.generator")

class LLMGenerator:
    def __init__(self):
        self.llm_provider = settings.LLM_PROVIDER.lower() if settings.LLM_PROVIDER else None
        self.client = None

        if self.llm_provider == "openai" and settings.OPENAI_API_KEY:
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info("✅ LLM Generator initialized with OpenAI")
        elif self.llm_provider == "anthropic" and settings.ANTHROPIC_API_KEY:
            self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
            logger.info("✅ LLM Generator initialized with Anthropic")
        elif self.llm_provider == "nvidia" and settings.NVIDIA_API_KEY:
            self.client = AsyncOpenAI(
                api_key=settings.NVIDIA_API_KEY,
                base_url=settings.NVIDIA_BASE_URL
            )
            logger.info(f"✅ LLM Generator initialized with NVIDIA ({settings.NVIDIA_LLM_MODEL})")
        else:
            logger.warning("⚠️ No valid LLM API key found or provider mismatch. Running in retrieval-only mode.")

    async def generate(self, query: str, context: str) -> str:
        """Non-streaming generation (fallback)"""
        if not self.client:
            return "LLM is disabled. Running in retrieval-only mode."

        try:
            if self.llm_provider == "openai" or self.llm_provider == "nvidia":
                model = settings.NVIDIA_LLM_MODEL if self.llm_provider == "nvidia" else "gpt-4o"
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": self._build_user_prompt(query, context)}
                    ],
                    temperature=1.0 if self.llm_provider == "nvidia" else 0.3,
                    max_tokens=4096 if self.llm_provider == "nvidia" else 1024
                )
                return response.choices[0].message.content.strip()

            elif self.llm_provider == "anthropic":
                response = await self.client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=1024,
                    temperature=0.3,
                    system=self._get_system_prompt(),
                    messages=[{"role": "user", "content": self._build_user_prompt(query, context)}]
                )
                return response.content[0].text.strip()

        except (APIError, AnthropicError) as e:
            logger.error(f"LLM API error: {e}")
            return f"Error generating answer: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected LLM error: {e}", exc_info=True)
            return "Sorry, I encountered an error while generating the answer."
    
    def generate_sync(self, query: str, context: str) -> str:
        """Synchronous wrapper for generate() - for non-async contexts"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a new one
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.generate(query, context))
                    return future.result()
            else:
                return loop.run_until_complete(self.generate(query, context))
        except Exception as e:
            logger.error(f"Sync generation error: {e}")
            return f"Error: {str(e)}"

    async def generate_stream(self, query: str, context: str) -> AsyncGenerator[str, None]:
        """Streaming generation - yields token by token (OpenAI/NVIDIA compatible)"""
        if not self.client:
            yield "LLM is disabled (retrieval-only mode). Context is ready above."
            return

        try:
            if self.llm_provider == "openai" or self.llm_provider == "nvidia":
                model = settings.NVIDIA_LLM_MODEL if self.llm_provider == "nvidia" else "gpt-4o"
                stream = await self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": self._build_user_prompt(query, context)}
                    ],
                    temperature=1.0 if self.llm_provider == "nvidia" else 0.3,
                    max_tokens=4096 if self.llm_provider == "nvidia" else 1024,
                    stream=True
                )

                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    
                    # Special check for NVIDIA Kimi reasoning content
                    reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                    if reasoning:
                        yield f"💭 {reasoning}"
                    
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

            elif self.llm_provider == "anthropic":
                async with self.client.messages.stream(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=1024,
                    temperature=0.3,
                    system=self._get_system_prompt(),
                    messages=[{"role": "user", "content": self._build_user_prompt(query, context)}]
                ) as stream:
                    async for text in stream.text_stream:
                        if text:
                            yield text

        except (APIError, AnthropicError) as e:
            logger.error(f"Streaming LLM error: {e}")
            yield f"\n\n[Error: {str(e)}]"
        except Exception as e:
            logger.error(f"Unexpected streaming error: {e}", exc_info=True)
            yield "\n\n[Sorry, an error occurred while streaming the answer.]"

    def _get_system_prompt(self) -> str:
        prompt = """You are a highly accurate and helpful coding assistant.
Use ONLY the provided context to answer the question.
Be precise, concise, and technical when appropriate.
If the context does not contain enough information, clearly state that."""
        
        if self.llm_provider == "nvidia":
            prompt = "You are Kimi, an AI assistant created by Moonshot AI. " + prompt
            
        return prompt

    def _build_user_prompt(self, query: str, context: str) -> str:
        return f"""Context:
{context}

Question: {query}

Answer:"""