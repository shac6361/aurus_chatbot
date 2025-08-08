import anthropic

client = anthropic.Anthropic(
    api_key="your-claude-key"
)

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",  # or other available models
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "What's in this image?"}
    ]
)

print(message.content[0].text)