import time
from openai import OpenAI, OpenAIError

client = OpenAI()

while True:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-eva",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        print(response.choices[0])
        break  # 请求成功，退出循环
    except OpenAIError as e:
        print(e)

