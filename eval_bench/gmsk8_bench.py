import json
import re
from tqdm import tqdm
from ollama import chat, ChatResponse
from rich.console import Console
from rich.table import Table

# -------------------------
# Configuration and Dataset
# -------------------------

MODEL_NAMES = ["qwen2-math:1.5b","Qwen2.5-GRPO-RL","Qwen2.5-Base"]

# Load GMSK8 test data from JSONL file
def load_test_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

TEST_DATA = load_test_data('/home/n/Token-Book/eval/Benchmark/grade-school-math/grade_school_math/data/test50.jsonl')

# -------------------------
# Helpers to extract answers
# -------------------------

def extract_correct_answer(text):
    """
    Extract the correct (gold) answer after '####' in the question's answer field.
    E.g. '... #### 18' -> 18
    """
    match = re.search(r"#### (\d+)", text)
    if match:
        return int(match.group(1))
    return None

def extract_numeric_answer(response):
    """
    Extract a numeric answer from the LLM response (best guess).
    This example simply grabs the last integer found in the response.
    """
    matches = re.findall(r"\b\d+\b", response)
    if matches:
        return int(matches[-1])  # pick the last number in the text
    return None

# -------------------------
# Query models
# -------------------------

def query_model(model_name, question, max_tokens=1500):
    """
    Send a question to the specified Ollama model and get the response.
    """
    response = chat(model=model_name, messages=[{"role": "user", "content": question}])
    
    # Extract the content of the response from the 'Message' object
    if isinstance(response, dict) and "message" in response:
        return response["message"]["content"]
    elif isinstance(response, ChatResponse):
        return response.message.content  # Correctly access the content of the response
    return str(response)  # In case the response is something else

# -------------------------
# Evaluation logic
# -------------------------

def evaluate_models(test_data):
    """
    Evaluate models on the test data and display the results in a scoreboard.
    Also saves detailed outputs to evaluation.json.
    """
    console = Console()
    
    # Prepare scoreboard
    scoreboard = {model: 0 for model in MODEL_NAMES}
    
    # We'll store detailed results from every query here
    evaluation_results = []

    # Prepare table
    table = Table(title="Model Evaluation Results")
    table.add_column("Model", justify="center")
    table.add_column("Correct Answers", justify="center")
    table.add_column("Total Questions", justify="center")
    
    # Evaluate each model
    for model_name in MODEL_NAMES:
        correct_count = 0
        
        # Evaluate each test case
        for data in tqdm(test_data, desc=f"Evaluating {model_name}"):
            question = data["question"]
            gold_answer = extract_correct_answer(data["answer"])
            
            # Query the model
            raw_response = query_model(model_name, question)
            
            # Extract a numeric answer from the LLM's response
            model_extracted_answer = extract_numeric_answer(raw_response)
            
            # Check correctness
            is_correct = (model_extracted_answer == gold_answer)
            if is_correct:
                correct_count += 1
            
            # Record full details for JSON output
            evaluation_results.append({
                "model": model_name,
                "question": question,
                "gold_answer": gold_answer,
                "raw_response": raw_response,
                "model_extracted_answer": model_extracted_answer,
                "is_correct": is_correct
            })
        
        # Update scoreboard
        scoreboard[model_name] = correct_count
        table.add_row(model_name, str(correct_count), str(len(test_data)))
    
    # Print scoreboard
    console.print(table)
    
    # Write detailed results to JSON
    with open("evaluation.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

# -------------------------
# Run the evaluation
# -------------------------

evaluate_models(TEST_DATA)

