from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch

# Specify the model name from Hugging Face Model Hub
model_name = "CohereForAI/aya-expanse-8b"  # Replace with your desired model

preamble = """
Here's an email that I wrote.
Can you make a table of all the proper nouns, such as names of things and projects, cities, people, etc.
used in this email and their numerical position in the word order and output it as a CSV. The columns of the 
CSV should be 'noun', 'type', and 'position in word order'. Make sure you only include the proper nouns.
Please only output the CSV. Here's the email:
"""


def load_and_preprocess_json(json_file):
   

    with open(json_file, 'r') as f:
        raw_json = [json.loads(line) for line in f.readlines()]
    return[j["email"] for j in raw_json]
    return[f"{preamble} {e['email']}" for e in raw_json]
    return[f"{preamble}  subject: {e['subject']}\n\nemail: {e['email']}" for e in raw_json]



def main(email_file):

    queries = load_and_preprocess_json(email_file)[:10]

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Run queries through the model
    # ExamVple queries
    responses = [annotate_text(query, model, tokenizer, device) for query in queries]
    for query, response in zip(queries, responses):
        print(f"Query: {query}")
        print(f"Response: {response}\n")

    values_dict = {}
    for response in responses:
        lines = response.split("\n")
        # Assume the first line contains the header.
        # Todo: test this.
        content_lines = lines[1:]
        for line in content_lines:
            items = line.split(",")
            if len(items) != 3:
                print("bad line", line)
                continue
            word, word_type, order = items
            if word in values_dict:
                values_dict[word]["count"] += 1
                values_dict[word]["type"].add(word_type)
            else:
                values_dict[word] = {"count": 1, "type": {word_type}}
    print(values_dict)
        



def annotate_text(query, model, tokenizer, device, max_length=500):
    messages = [{"role": "system", "content": preamble},
                {"role": "user", "content": query}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    input_ids = input_ids.to('cuda')
    gen_tokens = model.generate(
        input_ids,
        max_new_tokens=500,
        do_sample=False,
        )

    answer = tokenizer.decode(gen_tokens[0, input_ids.shape[1]:],skip_special_tokens=True)

    return answer



queries_file = "../data/david_anonymous_clean.jsonl"
main(queries_file)
