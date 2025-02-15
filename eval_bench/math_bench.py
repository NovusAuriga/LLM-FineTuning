import json
import re
from tqdm import tqdm
from ollama import chat, ChatResponse
from rich.console import Console
from rich.table import Table

# -------------------------
# Configuration and Dataset
# -------------------------

MODEL_NAMES = ["Qwen2.5-GRPO-RL"]

# Load GMSK8 test data from JSONL file
def load_test_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

TEST_DATA = load_test_data('/home/n/Token-Book/eval/Benchmark/grade-school-math/grade_school_math/data/test50.jsonl')

# Fixed generation options
FIXED_OPTIONS = {}

# Helper: Generate a list of values from start to end (inclusive) with a given step.
def generate_values(start, end, step):
    values = []
    current = start
    while current <= end:
        values.append(round(current, 2))
        current += step
    return values

# Define dynamic hyperparameter ranges.
TEMPERATURE_VALUES = generate_values(0.6, 0.6, 0.4)
TOP_P_VALUES = generate_values(0.9, 0.9, 0.1)
NUM_PREDICT_VALUES = generate_values(1500, 1500, 500)

# Number of iterations for the experiment.
NUM_ITERATIONS = 1

# -------------------------
# Debug Evaluation Function
# -------------------------

def evaluate_model_debug(model_name, test_data, options):
    """
    For each test sample, send the question to the model, print debugging info,
    and extract the final answer using various regex patterns. Additionally, if the
    response contains a boxed answer with extra content (e.g. "d) 3.5"), we extract
    the first option letter (a–e) found in the boxed content. If no letter is found,
    we then assume it is a numeric answer, normalize (i.e. remove spaces) both the
    boxed content and the expected answer from the test data, and compare them.
    We also count cases where the LLM appears to produce an "empty" answer (i.e. no
    alphanumeric characters).

    Returns a dict with:
      - "accuracy": percentage correct
      - "responses": a list with one dict per sample containing:
           - "question": the problem statement
           - "correct_answer": the expected answer (option letter)
           - "llm_answer": the extracted answer (or "none" if no valid answer was extracted)
           - "llm_last_chars": the last 80 characters of the cleaned LLM output
      - "empty_count": count of responses that contained no alphanumeric characters
    """
    correct_count = 0
    responses = []
    empty_count = 0  # Count of responses that are effectively empty

    for sample in tqdm(test_data, desc=None, leave=False, position=1):
        question = sample["question"]
        ref_answer = sample["answer"].split("####")[-1].strip()  # Extract the final part of the answer
        options_list = sample["answer"].split(", ")  # Modify this if GMSK8 has a different structure

        # Build a mapping from option letter to answer text.
        options_map = {}
        for opt in options_list:
            # Expected format: "a ) 38" (or similar)
            m = re.match(r'([a-eA-E])\s*\)\s*(.+)', opt)
            if m:
                letter = m.group(1).lower()
                answer_text = m.group(2).strip()
                options_map[letter] = answer_text

        try:
            chat_options = options.copy()
            chat_options["choices"] = options_list

            response: ChatResponse = chat(
                model=model_name,
                messages=[
                    {"role": "user", "content": (
                        "<think>\nYou are a reasoning assistant. Please reason through the problem step by step.\n"
                        "Here are the multiple-choice options for this question:\n"
                        f"{options_list}\n\n"
                        "Question: " + question + "\n\n"
                        "Your task is to carefully analyze the question, evaluate each option, and select the most appropriate answer from the options.\n"
                        "Please provide your answer in the following format: 'answer: x' (where 'x' is the letter of the correct choice, e.g., 'answer: a'),\n"
                        "or if you think the numerical solution is more appropriate, you may provide it inside a boxed format like '\\boxed{38}'.\n"
                    )}
                ],
                options=chat_options
            )
            predicted = response.message.content.strip()

        except Exception as e:
            predicted = f"Error: {e}"

        # Normalize whitespace and remove extra asterisks.
        predicted_clean = re.sub(r'\s+', ' ', predicted.strip())
        predicted_clean = re.sub(r'\*', '', predicted_clean)

        # Check for an "empty" response (no alphanumeric characters).
        if not re.search(r'[a-zA-Z0-9]', predicted_clean):
            empty_count += 1

        # We'll consider the last 80 characters for our extraction.
        last_chars = predicted_clean[-80:]

        answer_found = False
        final_answer_extracted = ""

        # --- Attempt to extract from a boxed format ---
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", last_chars.strip())
        if boxed_match:
            boxed_content = boxed_match.group(1).strip()
            # First, search for any letter (a–e) in the boxed content.
            letter_match = re.search(r'([a-eA-E])', boxed_content)
            if letter_match:
                final_answer_extracted = letter_match.group(1).lower()
                answer_found = True
            else:
                # Otherwise, assume it is a numeric answer.
                try:
                    # Remove all spaces from the boxed content.
                    numeric_str_boxed = re.sub(r'\s+', '', boxed_content)
                    boxed_number = float(numeric_str_boxed)
                    # If the number matches the expected answer, accept it.
                    if abs(boxed_number - float(ref_answer)) < 1e-6:
                        final_answer_extracted = ref_answer
                        answer_found = True
                    else:
                        final_answer_extracted = "none"
                except Exception as e:
                    final_answer_extracted = "none"

        # --- Fallback extraction using a primary pattern ---
        if not answer_found:
            pattern = re.compile(r'(?i)(?:\*{0,2}answer|\*{0,2}onepiece\s*|answer\s*is).*?([a-e])')
            match = pattern.search(last_chars)
            if match:
                final_answer_extracted = match.group(1).lower()
                answer_found = True

        # --- Another fallback: look for a letter followed by a closing parenthesis, e.g., "e )" ---
        if not answer_found:
            letter_with_paren = re.search(r'([a-eA-E])\s*\)', last_chars.strip())
            if letter_with_paren:
                final_answer_extracted = letter_with_paren.group(1).lower()
                answer_found = True

        # If no valid answer is extracted, record "none" as the llm_answer.
        llm_answer = final_answer_extracted if final_answer_extracted else "none"

        if llm_answer == ref_answer:
            is_correct = True
        else:
            is_correct = False

        if is_correct:
            correct_count += 1

        responses.append({
            "question": question,
            "correct_answer": ref_answer,
            "llm_answer": llm_answer,
            "llm_last_chars": last_chars,
            "parsed_options": options_map  # Useful for debugging.
        })
    
    accuracy = (correct_count / len(test_data)) * 100 if test_data else 0
    return {"accuracy": accuracy, "responses": responses, "empty_count": empty_count}

# -------------------------
# Main Evaluation Loop and Final Average Calculation
# -------------------------

def main():
    console = Console()
    overall_iterations = []  # To store each iteration's detailed results.
    iteration_results_list = []  # To accumulate per-iteration nested accuracy results.
    all_responses = []         # To accumulate all individual sample responses.
    
    # Accumulate empty response counts per model across evaluations.
    empty_counts_by_model = {model: 0 for model in MODEL_NAMES}

    total_evals = (len(MODEL_NAMES) * len(TOP_P_VALUES) *
                   len(TEMPERATURE_VALUES) * len(NUM_PREDICT_VALUES) * NUM_ITERATIONS)
    progress_bar = tqdm(total=total_evals, desc="Evaluating combinations", unit="eval", position=0)

    console.print("[bold blue]Running debugging iterations over Top-p, Temperature, and num_predict[/bold blue]")
    for iteration in range(1, NUM_ITERATIONS + 1):
        console.print(f"[bold]Iteration {iteration}[/bold]")
        iteration_results = {}  # Structure: { model: { top_p: { temperature: { num_predict: accuracy } } } }

        for model in MODEL_NAMES:
            iteration_results[model] = {}
            for tp in TOP_P_VALUES:
                iteration_results[model][tp] = {}
                for temp in TEMPERATURE_VALUES:
                    iteration_results[model][tp][temp] = {}
                    for num_predict in NUM_PREDICT_VALUES:
                        opts = {"temperature": temp, "top_p": tp, "num_predict": num_predict}
                        opts.update(FIXED_OPTIONS)
                        result = evaluate_model_debug(model, TEST_DATA, opts)
                        iteration_results[model][tp][temp][num_predict] = result["accuracy"]
                        all_responses.extend(result["responses"])
                        empty_counts_by_model[model] += result["empty_count"]
                        progress_bar.update(1)
        iteration_results_list.append(iteration_results)
        overall_iterations.append(iteration_results)

        # For debugging, print a table of accuracy per model.
        for model in MODEL_NAMES:
            table = Table(title=f"Iteration {iteration} - {model} (Top-p vs Temperature vs num_predict)")
            table.add_column("Top-p", justify="center", style="magenta")
            headers = []
            for temp in TEMPERATURE_VALUES:
                for num_predict in NUM_PREDICT_VALUES:
                    headers.append(f"{temp:.2f}/{num_predict}")
            for header in headers:
                table.add_column(header, justify="center", style="cyan")
            for tp in sorted(TOP_P_VALUES):
                row = [f"{tp:.2f}"]
                for temp in TEMPERATURE_VALUES:
                    for num_predict in NUM_PREDICT_VALUES:
                        acc = iteration_results[model][tp][temp].get(num_predict, None)
                        acc_str = f"{acc:.2f}" if acc is not None else "-"
                        row.append(acc_str)
                table.add_row(*row)
            console.print(table)
            console.print("\n")
        console.print("=" * 60 + "\n")
    progress_bar.close()

    # Compute the final overall average for each model grouped by each num_predict value.
    overall_avg_by_np = {}
    for model in MODEL_NAMES:
        overall_avg_by_np[model] = {}
        for num_predict in NUM_PREDICT_VALUES:
            overall_avg_by_np[model][num_predict] = []
        for iteration_results in iteration_results_list:
            for tp in TOP_P_VALUES:
                for temp in TEMPERATURE_VALUES:
                    for num_predict in NUM_PREDICT_VALUES:
                        acc = iteration_results[model][tp][temp].get(num_predict)
                        if acc is not None:
                            overall_avg_by_np[model][num_predict].append(acc)
        for num_predict in NUM_PREDICT_VALUES:
            scores = overall_avg_by_np[model][num_predict]
            overall_avg_by_np[model][num_predict] = sum(scores) / len(scores) if scores else 0.0

    final_table = Table(title="Final Overall Average Accuracy by num_predict (per Model)")
    final_table.add_column("num_predict", justify="center", style="magenta")
    for model in MODEL_NAMES:
        final_table.add_column(model, justify="center", style="cyan")
    for num_predict in NUM_PREDICT_VALUES:
        row = [str(num_predict)]
        for model in MODEL_NAMES:
            avg_acc = overall_avg_by_np[model].get(num_predict, 0.0)
            row.append(f"{avg_acc:.2f}")
        final_table.add_row(*row)
    console.print(final_table)

    # Print a table for the empty response counts per model.
    empty_table = Table(title="Empty Response Counts per Model")
    empty_table.add_column("Model", justify="center", style="magenta")
    empty_table.add_column("Empty Responses", justify="center", style="cyan")
    for model in MODEL_NAMES:
        empty_table.add_row(model, str(empty_counts_by_model[model]))
    console.print(empty_table)

    final_results = {
        "iterations": overall_iterations,
        "overall_avg_by_num_predict": overall_avg_by_np,
        "empty_counts_by_model": empty_counts_by_model
    }
    with open("evaluation_iterations_debug.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    with open("evaluation_math.json", "w", encoding="utf-8") as f:
        json.dump(all_responses, f, indent=2, ensure_ascii=False)

    console.print("[bold green]Debug evaluation complete. Detailed results saved to evaluation_iterations_debug.json and evaluation_math.json[/bold green]")

if __name__ == "__main__":
    main()

