# RL Principles
import json
from llama_cpp import Llama
import pandas as pd

# Preference selection loop

# Replacing preferences system with scoring

def ask_prompt(prompt):
    response=lcpp_llm(prompt=prompt, max_tokens=512, temperature=1, top_p=0.95,
                  repeat_penalty=1.2, top_k=150,
                  echo=True)
    return response

def ai_evaluate(answers, question, principle):
    system_prompt="SYSTEM: You are the ASSISTANT. You only take part in this conversation as the ASSISTANT. Respond concisely and short.\n"
    prompt = system_prompt + "Consider the following question:\nHUMAN: " + question + "\n\n" + principle + "\n" + answers + "\nSYSTEM: Please answer only by saying \"Option 1\" or \"Option 2\".\n\nAssistant: "
    response = ask_prompt(prompt)
    print("----Response: ", response["choices"][0]["text"][-1])
    return response["choices"][0]["text"][-1]

def generate_preference(initial_prompt):

    with open('RLMadisonInstructions.json') as json_file:
    # Load the JSON data
        principles = json.load(json_file)
    
    ai_generated_data = []

    # generating initial responses
    response1 = ask_prompt(initial_prompt)
    response2 = ask_prompt(initial_prompt)

    r1_text = response1["choices"][0]["text"].replace(initial_prompt, "")
    r2_text = response2["choices"][0]["text"].replace(initial_prompt, "")


    ai_generated_data = [initial_prompt, r1_text, r2_text]

    # Answers
    answers = f"\n1. \"{r1_text}\"\n2. \"{r2_text}\"\n"

    # Get AI evaluation based on constitutional principles (replace this with your actual AI evaluation logic)
    for i in principles :
        ai_preference = ai_evaluate(answers, initial_prompt, i)
        # clean up preference value
        pref = ai_preference["choices"][0]["text"].replace(r1_text, "")
        pref = pref.replace(r2_text, "")
        pref = pref.replace(initial_prompt, "")
        pref = pref.replace(i,"")
        ai_generated_data.append(pref)

    # Create a data point for the AI-generated preference dataset
    print("appended data points: ", ai_generated_data)

    return ai_generated_data




def main():
    df = pd.DataFrame(columns=['Q', 'A1', 'A2', 'V1', 'V2', 'V3', 'V4'])
    with open('questions_clean.json') as f:
        questions = json.load(f)
        for i in questions:
            row = generate_preference(i)
            df = df.append(row, ignore_index=True)

if __name__ == "__main__":
    main()