# WhatsApp Sentiment Analyzer (Baby Formula Market Research)

This application is a specialized data processing pipeline designed to analyze sentiment in WhatsApp chat logs, specifically focused on the baby formula market.

It automates the process of ingesting raw chat exports, filtering messages based on brand-specific keywords, and utilizing Large Language Models (LLMs) via the POE API to determine sentiment (Positive, Negative, Neutral) and extract reasoning.

## 🌟 Features

*   **Data Validation:** Uses `Pandera` and `Pydantic` to ensure input data (keywords and chat logs) and LLM outputs strictly adhere to defined schemas.
*   **Keyword Filtering:** Intelligent pre-filtering of messages based on brand keywords and "required product" logic to reduce LLM costs and noise.
*   **Asynchronous Processing:** Utilizes `asyncio` and `AsyncOpenAI` for concurrent LLM requests, significantly speeding up the analysis of large datasets.
*   **Robust AI Interaction:** Includes retry logic for failed API calls and auto-correction prompts if the LLM returns invalid JSON.
*   **Data Preparation Tools:** Includes utilities (`merger.py`) to merge, clean, deduplicate, and sort raw CSV chat logs.

## 📂 Project Structure

*   **`main.py`**: The entry point of the application. Orchestrates the loading, processing, and saving of data.
*   **`utils/`**
    *   **`ai.py`**: Handles interactions with the LLM provider (POE). Manages system prompts and parses/validates JSON responses.
    *   **`chatprocessor.py`**: Core logic for tagging keywords in dataframes and managing the async sentiment analysis loop.
    *   **`loader.py`**: Simple wrappers for loading Excel and CSV files.
    *   **`merger.py`**: Utility script to merge scattered CSV files, remove duplicates, and sort by date/time.
    *   **`preprocessor.py`**: Handles loading chat folders, combining files, and validating data against schemas.
    *   **`validator.py`**: Defines `Pandera` schemas for DataFrames and `Pydantic` models for AI responses.

## 🚀 Setup & Installation

### Prerequisites
*   Python 3.10+
*   A POE API Key (or compatible OpenAI-format provider)

### Installation

1.  **Clone the repository**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Install dependencies**
    ```bash
    pip install pandas pandera pydantic openai tqdm python-dotenv openpyxl
    ```

3.  **Environment Configuration**
    Create a `.env` file in the root directory and add your API key:
    ```env
    POE_API_KEY=your_actual_api_key_here
    ```

## 📊 Input Data Format

The application expects data in a specific `data/` directory structure.

### 1. Keywords File (`data/keywords.xlsx`)
An Excel file defining which brands to look for.
**Columns:**
*   `brand`: The main brand name.
*   `product`: Specific product line.
*   `keyword`: The keyword to search for in messages.
*   `required_product`: (Optional) A dependency. The keyword is only valid if the message *also* contains a keyword associated with this product.

### 2. Chat Logs (`data/chats/`)
The folder structure should be:
```text
data/
  chats/
    Group_Name_A/
      chat_log_part1.csv
      chat_log_part2.csv
    Group_Name_B/
      chat_log.csv
```

**CSV Schema (Required Columns):**
*   `Date2` (Date)
*   `Time`
*   `userPhone`
*   `messageBody`
*   `quotedMessage` (Optional)
*   `mediaType` (Optional)
*   `mediaCaption` (Optional)

## 🛠 Usage

### Step 1: Data Preparation (Optional)
If you have raw, scattered CSV files, you can use the merger tool to clean them up before analysis.
```bash
python utils/merger.py
```
*This will look in `../data/to_merge`, merge files by name, deduplicate, sort, and save to `../data/merged`.*

### Step 2: Run Analysis
To run the main sentiment analysis pipeline:
```bash
python main.py
```

**What happens during execution:**
1.  **Preprocessing:** The app reads `keywords.xlsx` and iterates through folders in `data/chats`.
2.  **Tagging:** It creates columns for every brand found in the keyword file. It marks rows with `1` if a keyword is found in `messageBody`.
3.  **Analysis:** For every row marked with a keyword match, it sends the message to the LLM (Gemini-2.5-flash via POE).
4.  **Output:** The results are saved to `data/output.xlsx`, with separate sheets for each chat group. The output includes the original data plus:
    *   Sentiment columns (P/N/I) for each brand.
    *   A `Reason` column containing the AI's explanation (in Traditional Chinese).

## 🧠 AI & Prompts

The logic in `utils/ai.py` configures the AI as a "Market Research Analyst".
*   **Model:** Defaults to `gemini-2.5-flash` (configurable in `main.py`).
*   **Sentiment Logic:**
    *   **P (Positive):** Praise, purchase intent, positive health effects.
    *   **N (Negative):** Complaints, side effects (constipation, allergies), high price.
    *   **I (Neutral):** General inquiries, factual statements.
*   **Validation:** The system uses `Pydantic` to enforce that the LLM returns valid JSON. If the LLM returns malformed data, the system automatically feeds the error back to the LLM and asks for a correction (up to 3 retries).
