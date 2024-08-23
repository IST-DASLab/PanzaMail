from hosting import PanzaWebService
from llm import OllamaLLM, ChatHistoryType

llm = OllamaLLM("custom", "path/to/file", {})

messages: ChatHistoryType = [{"role": "user", "content": "Write a one-sentence email saying i will be late to the meeting"}]

# Example of how to use the LLM with no streaming
stream = llm.chat_stream(messages)
while True:
    try:
        print(next(stream))
    except StopIteration:
        break
    
# Example of how to use the LLM with streaming
print(llm.chat(messages))

# create a new PanzaWebService
service = PanzaWebService()