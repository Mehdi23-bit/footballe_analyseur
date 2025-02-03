from openai import OpenAI
import sys
import os

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.environ["api_key"],
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