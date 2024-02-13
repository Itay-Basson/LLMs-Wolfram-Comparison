
# LLM Comparison Tool

This Python script evaluates and compares the performance of different Large Language Models (LLMs) against answers provided by the Wolfram Alpha API. It is designed to query multiple models for answers to a set of predefined questions, compare these answers for correctness, and visualize the results.

## Features

- Retrieves questions from a CSV file and queries each question against the Wolfram Alpha API and selected LLMs.
- Compares the LLMs' answers with Wolfram Alpha's answers to evaluate correctness.
- Caches answers using Redis to optimize performance.
- Visualizes the comparison results, showing the effectiveness and response times of each model.

## Getting Started

### Prerequisites

- Python 3.6+.
- Redis server running locally or accessible remotely.
- Access to the Wolfram Alpha API. You'll need to sign up for an API key.
- Access to GPT4All API or library. Ensure you have the necessary permissions or API keys to use GPT4All for querying LLMs.

### Installation

1. **Clone the Repository**

```bash
git clone https://github.com/your_github_username/llm-comparison-tool.git
cd llm-comparison-tool
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Configure the Application**

Edit `config.py` to set up your environment:

- `WOLFRAM_ALPHA_APP_ID`: Your API key for Wolfram Alpha.
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`: Configuration for your Redis server.
- `CSV_FILE_PATH`: Path to your CSV file containing the questions.

### Usage

1. **Prepare Your Questions CSV File**

Ensure your CSV file is formatted with at least two columns: `Category` and `Question`.

2. **Run the Script**

```bash
python main.py
```

3. **View the Results**

The script prints the comparison results in the console and generates visualizations for further analysis.

### Customizing LLM Models

The script currently supports querying specific LLMs but can be easily modified to include additional models or change existing ones:

- To modify or add LLM models, update the `get_answer_from_llm_1` and `get_answer_from_llm_2` functions in `main.py`. Specify the model identifier and adjust parameters as needed.

```python
# Example modification for a new model
def get_answer_from_new_model(question, max_tokens):
    model = GPT4All("new-model-identifier")
    # Further implementation...
```

### Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to improve the script or add new features.

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments

- Thanks to the Wolfram Alpha API for providing computational answers.
- Thanks to the developers of the GPT models and the Redis database for their outstanding tools.
