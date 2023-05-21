import json
import openai

with open('./data/sample.jsonl', 'r') as f:
    prompts = f.readlines()

prompts = [json.loads(prompt) for prompt in prompts]
sample_prompt = prompts[7]['messages'][1]['content']
print(sample_prompt)
messages = [{
    "role": "system",
    "content": "You are a helpful assistant."
}, {
    "role": "user",
    "content": sample_prompt
}]
response = openai.ChatCompletion.create(messages=messages,
                                        model='gpt-3.5-turbo',
                                        temperature=0.7,
                                        top_p=1)
print(f"Finish reason: {response['choices'][0]['finish_reason']}")
print(response['choices'][0]['message']['content'])
breakpoint()