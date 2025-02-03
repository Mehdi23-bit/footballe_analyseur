from openai import OpenAI
import sys

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

completion = client.chat.completions.create(
#   extra_headers={
#     "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
#     "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
#   },
  model="deepseek/deepseek-r1-distill-qwen-1.5b",
  messages=[
    {
      "role": "user",
      "content": "hello deep seek how are you "
    }
  ]
)
print(completion.choices[0].message.content)