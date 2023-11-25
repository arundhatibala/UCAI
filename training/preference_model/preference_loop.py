# RL Principles
import json
from llama_cpp import Llama

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
    #print("----Final Prompt:", prompt)
    response = ask_prompt(prompt)
    print("----Response: ", response["choices"][0]["text"])
    return response

def rlaif_stage(initial_prompt, principle, question):
    ai_generated_data = []
    response1 = ask_prompt(initial_prompt)
    response2 = ask_prompt(initial_prompt)
    #print("----initial responses generated")

    r1_text = response1["choices"][0]["text"].replace(initial_prompt, "")
    r2_text = response2["choices"][0]["text"].replace(initial_prompt, "")

    #print("----r1text: " + r1_text)

    # Answers
    answers = f"\n1. \"{r1_text}\"\n2. \"{r2_text}\"\n"
    #print("----answers: " + answers)

    # Get AI evaluation based on constitutional principles (replace this with your actual AI evaluation logic)
    ai_preference = ai_evaluate(answers, question, RL_inst[0])
    print("preferences generated")

    # clean up preference value
    pref = ai_preference["choices"][0]["text"].replace(r1_text, "")
    pref = pref.replace(r2_text, "")
    pref = pref.replace(initial_prompt, "")
    pref = pref.replace(RL_inst[0],"")

    # Create a data point for the AI-generated preference dataset
    ai_generated_data.append({'prompt': initial_prompt, 'response1': (r1_text), 'response2': (r2_text), 'ai_preference': pref})
    print("appended data points")

    return ai_generated_data




def main():
    with open('RLMadisonInstructions.json') as json_file:
        # Load the JSON data
        RL_principles = json.load(json_file)

if __name__ == "__main__":
    main()