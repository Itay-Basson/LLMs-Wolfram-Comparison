from gpt4all import GPT4All
import wolframalpha
import csv
import redis
import time
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import re
from config import WOLFRAM_ALPHA_APP_ID, REDIS_HOST, REDIS_PORT, REDIS_DB, CSV_FILE_PATH, LLM_MODELS



def read_questions_from_csv(file_path):
    """
        Reads questions and categories from a CSV file specified by file_path.

        Args:
        file_path (str): The path to the CSV file.

        Returns:
        list of tuples: Each tuple contains (category, question) extracted from the CSV.
        """
    question_category_pairs = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Read the header row
        category_index = header.index('Category')  # Find the index of the 'Category' column
        question_index = header.index('Question')  # Find the index of the 'Question' column
        for row in reader:
            category = row[category_index]  # Get the category using the found index
            question = row[question_index]  # Get the question using the found index
            question_category_pairs.append((category, question))
    return question_category_pairs


# Replace 'path_to_your_csv.csv' with the actual path to your CSV file
questions = read_questions_from_csv(CSV_FILE_PATH)

try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
except Exception as e:
    print(f"Failed to connect to Redis: {e}")
    exit(1)  # Exit if Redis connection fails


# Initialize Wolfram Alpha client
wolfram_client = wolframalpha.Client(WOLFRAM_ALPHA_APP_ID)


def get_answer_from_wolfram(question, redis_client):
    # Try to get the answer from Redis cache first
    cached_answer = redis_client.get(question)
    if cached_answer:
        print(f"Retrieved from cache: {question}")
        return cached_answer.decode('utf-8')  # Decode byte string to string

    try:
        # If not cached, query Wolfram Alpha
        print(f"Querying Wolfram Alpha API for: {question}")
        res = wolfram_client.query(question)
        answer = next(res.results).text

        # Cache this new answer in Redis for 4 hours (14400 seconds)
        redis_client.setex(question, 14400, answer)
        return answer
    except Exception as e:
        print(f"Error getting answer for question '{question}': {e}")
        return None


def get_answer_from_llm_1(question, max_tokens):
    # This function queries the Orca-Mini model for an answer.
    # Model identifier can be modified to use a different model if needed.
    model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")
    start_time = time.time()  # Start timing
    output = model.generate(question, max_tokens)
    end_time = time.time()  # End timing
    time_elapsed_ms = int((end_time - start_time) * 1000)  # Convert time to milliseconds
    return output, time_elapsed_ms

def get_answer_from_llm_2(question, max_tokens):
    # This function queries the Replit model for an answer.
    # Model identifier can be modified to use a different model if needed.
    # 'temp=0.1' is a temperature setting that can be adjusted for different response characteristics.
    model = GPT4All("replit-code-v1_5-3b-q4_0.gguf")
    start_time = time.time()  # Start timing
    output = model.generate(question, max_tokens, temp=0.1)
    end_time = time.time()  # End timing
    time_elapsed_ms = int((end_time - start_time) * 1000)  # Convert time to milliseconds
    return output, time_elapsed_ms




def compare_answers(question, wolfram_answer, llm_answer):
    # Constructing the prompt for similarity comparison
    prompt = (f"### User:"               
             f"Calculate the similarity score between these two answers on a scale of 0.0 to 1.0 and"              
             f"Provide the similarity score as a decimal number WITHOUT additional commentary:" 
             f"\nAnswer 1: {llm_answer}\nAnswer 2: {wolfram_answer}\n" 
             f"The similarity score between the Answers is: "
             )

    # Using the first LLM to evaluate similarity
    similarity_response, response_time = get_answer_from_llm_1(prompt, max_tokens=5)
    #print(f"LLM Response: {similarity_response}, Response Time: {response_time}ms")

    # Try to extract a numeric similarity score
    try:
        # Remove parentheses and extract only the numerical part
        similarity_score_str = re.sub(r'[^\d.]+', '', similarity_response.split('\n')[0])
        similarity_score = float(similarity_score_str.strip())
    except ValueError:
        # If the response is not a valid float, use a default value (e.g., 0.0)
       # print(f"Invalid similarity score, using default value. Response was: {similarity_response}")
        similarity_score = 0.0  # Default value

    return similarity_score


# Function to print a single result
def print_result(result):
    print(f"Category: {result['Category']}")
    print(f"Question: {result['Question']}")
    print(f"Model: {result['Model']}")
    print(f"Answer: {result['Answer']}")
    print(f"Time Taken (ms): {result['TimeInMillisecondsToGetAnswer']}")
    print(f"Correctness: {result['Correctness']}")
    print("-" * 50)  # Print a separator for readability

# Initialize the results list
results = []

for category, question in questions:
    # Get Wolfram Alpha's answer
    wolfram_answer = get_answer_from_wolfram(question, redis_client)
    if wolfram_answer is None:
        continue  # Skip the question if no answer from Wolfram Alpha

    # Get answers from both LLMs along with the time taken
    llm1_answer, time_llm1 = get_answer_from_llm_1(question, max_tokens=12)
    llm2_answer, time_llm2 = get_answer_from_llm_2(question, max_tokens=12)

    # Calculate correctness for LLM1
    correctness_llm1 = compare_answers(question, wolfram_answer, llm1_answer)

    # Calculate correctness for LLM2
    correctness_llm2 = compare_answers(question, wolfram_answer, llm2_answer)

    # Add LLM1 results to the list and print
    llm1_result = {
        'Category': category,
        'Question': question,
        'Model': 'Orca-Mini',
        'Answer': llm1_answer,
        'TimeInMillisecondsToGetAnswer': time_llm1,
        'Correctness': float(correctness_llm1)
    }
    results.append(llm1_result)
    print_result(llm1_result)

    # Add LLM2 results to the list and print
    llm2_result = {
        'Category': category,
        'Question': question,
        'Model': 'Replit',
        'Answer': llm2_answer,
        'TimeInMillisecondsToGetAnswer': time_llm2,
        'Correctness': float(correctness_llm2)
    }
    results.append(llm2_result)
    print_result(llm2_result)

print("-" * 50)
print("-" * 50)
print("-" * 50)
print("-" * 50)
print("-" * 50)
print("-" * 50)
print("-" * 50)
print("-" * 50)
print("-" * 50)
print("-" * 50)

def print_summary(results, llm1_name, llm2_name):
    total_questions_answered = 0
    total_rating_llm1 = 0
    total_rating_llm2 = 0
    lowest_rating_llm1 = float('inf')
    lowest_rating_llm2 = float('inf')
    lowest_rated_answer_llm1 = ""
    lowest_rated_answer_llm2 = ""
    lowest_rated_question_llm1 = ""
    lowest_rated_question_llm2 = ""

    for result in results:
        if result['Model'] == llm1_name:
            total_rating_llm1 += result['Correctness']
            if result['Correctness'] < lowest_rating_llm1:
                lowest_rating_llm1 = result['Correctness']
                lowest_rated_answer_llm1 = result['Answer']
                lowest_rated_question_llm1 = result['Question']
        elif result['Model'] == llm2_name:
            total_rating_llm2 += result['Correctness']
            if result['Correctness'] < lowest_rating_llm2:
                lowest_rating_llm2 = result['Correctness']
                lowest_rated_answer_llm2 = result['Answer']
                lowest_rated_question_llm2 = result['Question']
        total_questions_answered += 1

    avg_rating_llm1 = total_rating_llm1 / (total_questions_answered / 2)
    avg_rating_llm2 = total_rating_llm2 / (total_questions_answered / 2)

    print(f"Number of questions answered: {total_questions_answered // 2}")
    print(f"Average answer rating of {llm1_name}: {avg_rating_llm1:.2f}")
    print(f"Average answer rating of {llm2_name}: {avg_rating_llm2:.2f}")
    print(f"Lowest rating question and answer of {llm1_name} : {lowest_rated_question_llm1}")
    print(f"{lowest_rated_answer_llm1}")
    print(f"Lowest rating question and answer of {llm2_name} : {lowest_rated_question_llm2}")
    print(f"{lowest_rated_answer_llm2}")


def plot_correctness_vs_response_time(results, llm_name):
    # Extracting correctness scores and response times for the specified model
    correctness_scores = [res['Correctness'] for res in results if res['Model'] == llm_name]
    response_times = [res['TimeInMillisecondsToGetAnswer'] for res in results if res['Model'] == llm_name]

    plt.scatter(response_times, correctness_scores, alpha=0.5)
    plt.title(f'Correctness vs. Response Time for {llm_name}')
    plt.xlabel('Response Time (milliseconds)')
    plt.ylabel('Correctness Score')
    plt.grid(True)
    plt.show()


print_summary(results, 'Orca-Mini', 'Replit')


def plot_average_correctness_by_category_enhanced(results):
    # Initialize a dictionary to hold total correctness scores and counts
    correctness_totals = {}  # Structure: {('Category', 'Model'): [total_correctness, count]}

    for entry in results:
        key = (entry['Category'], entry['Model'])
        if key not in correctness_totals:
            correctness_totals[key] = [0, 0]  # Initialize total_correctness and count

        correctness_totals[key][0] += entry['Correctness']  # Add to total correctness
        correctness_totals[key][1] += 1  # Increment count

    # Calculate average correctness
    average_correctness = {k: total / count for k, (total, count) in correctness_totals.items()}

    # Separate the data for each LLM
    categories = sorted(set([k[0] for k in average_correctness.keys()]))  # Sorted list of categories
    data_llm1 = [average_correctness.get((cat, 'Orca-Mini'), 0) for cat in categories]
    data_llm2 = [average_correctness.get((cat, 'Replit'), 0) for cat in categories]

    # Use a more visually appealing style
    plt.style.use('ggplot')

    # Plotting the data with enhanced visuals
    x = range(len(categories))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, data_llm1, width, label='Orca-Mini', color='skyblue')
    rects2 = ax.bar([p + width for p in x], data_llm2, width, label='Replit', color='salmon')

    # Customizing the graph
    ax.set_ylabel('Average Correctness', fontsize=12)
    ax.set_title('Average Correctness by Category and Model', fontsize=14)
    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(categories, rotation=45)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.legend()

    ax.bar_label(rects1, padding=3, fontsize=10)
    ax.bar_label(rects2, padding=3, fontsize=10)

    # Adjust layout and spacing
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Call the function with your results
plot_average_correctness_by_category_enhanced(results)



# Example usage
def main():
    # Assuming 'results' is already populated with the data
    plot_correctness_vs_response_time(results, 'Orca-Mini')
    plot_correctness_vs_response_time(results, 'Replit')

if __name__ == "__main__":
    main()



