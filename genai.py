import os
import pandas as pd
import json
import google.generativeai as genai

def perform_genai_sentiment_analysis():
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GLOG_minloglevel"] = "2"

    # Fetch the API key from environment variables
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

    if GOOGLE_API_KEY is None:
        raise ValueError("API key not found! Please set the 'GOOGLE_API_KEY' environment variable.")

    # Configure your API with the key
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Load reviews from CSV file
    reviews_df = pd.read_csv('./backend/reviews.csv', header=None, names=['review'])

    # Initialize pred_label column
    reviews_df['label'] = None

    # Convert the DataFrame to JSON
    json_data = reviews_df.to_json(orient='records')

    prompt = f"""
    You are an expert linguist, who is good at classifying customer review sentiments into Positive/Negative labels.
    Help me classify customer reviews into: Positive(label=1) and Negative(label=0)
    Customer reviews are provided in the JSON format below.
    In your output, only return the JSON code back as output, which is provided between three backticks.
    Your task is to update predicted labels under 'label' in the JSON code.
    Don't make any changes to the JSON code format, please.

    ```
    {json_data}
    ```
    """

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # Debugging: Print out the raw response
        print("Raw API Response:", response_text)

        if not response_text:
            raise ValueError("Received an empty response from the API.")

        # Clean the data by stripping the backticks
        if response_text.startswith("```") and response_text.endswith("```"):
            response_text = response_text[3:-3].strip()
        elif response_text.startswith("`") and response_text.endswith("`"):
            response_text = response_text[1:-1].strip()

        # Debugging: Print out the cleaned response
        print("Cleaned API Response:", response_text)

        # Load the cleaned data and convert to DataFrame
        try:
            data = json.loads(response_text)
            df_sample = pd.DataFrame(data)

            # Save to CSV
            output_file = './dataset.csv'
            df_sample.to_csv(output_file, index=False)
            return output_file
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON from the response: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

