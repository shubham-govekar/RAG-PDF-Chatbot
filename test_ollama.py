import ollama

response = ollama.list()
print("Response:", response)
print("Type:", type(response))

if isinstance(response, dict):
    print("Keys:", response.keys())
    if 'models' in response:
        print("Models:", response['models'])