export OPENAI_API_KEY="1790715889671905303"
export OPENAI_BASE_URL="https://aigc.sankuai.com/v1/openai/native/chat/completions"

curl -X POST $OPENAI_BASE_URL \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-eva",
    "messages": [{"role": "user", "content": "你好！"}]
  }'
