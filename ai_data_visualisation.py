import re
import warnings
from typing import Optional, List, Any, Tuple
import streamlit as st
import pandas as pd
from together import Together

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

def match_code_blocks(llm_response: str) -> str:
    match = pattern.search(llm_response)
    if match:
        code = match.group(1)
        return code
    return ""

def chat_with_llm(user_message: str, dataset_path: str) -> Tuple[Optional[List[Any]], str]:
    # Update system prompt to include dataset path information
    system_prompt = f"""You're a Python data scientist and data visualization expert. You are given a dataset at path '{dataset_path}' and also the user's query.
You need to analyze the dataset and answer the user's query with a response and you run Python code to solve them.
IMPORTANT: Always use the dataset path variable '{dataset_path}' in your code when reading the CSV file."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    with st.spinner('Getting response from Together AI LLM model...'):
        client = Together(api_key=st.session_state.together_api_key)
        response = client.chat.completions.create(
            model=st.session_state.model_name,
            messages=messages,
        )

        response_message = response.choices[0].message
        python_code = match_code_blocks(response_message.content)
        
        if python_code:
            return python_code, response_message.content
        else:
            st.warning(f"Failed to match any Python code in model's response")
            return None, response_message.content

def upload_dataset(uploaded_file) -> str:
    dataset_path = f"./{uploaded_file.name}"
    try:
        with open(dataset_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return dataset_path
    except Exception as error:
        st.error(f"Error during file upload: {error}")
        raise error

def main():
    """Main Streamlit application."""
    # Add custom CSS styles to enhance the look
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background: linear-gradient(45deg, #FF416C, #FF4B2B);
            color: white;
            font-weight: bold;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 10px 20px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(45deg, #FF4B2B, #FF416C);
        }
        .stTextArea>textarea {
            background-color: rgba(255, 255, 255, 0.1);
            color: white;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stTextInput>input {
            background-color: rgba(255, 255, 255, 0.1);
            color: white;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stTitle, .stHeader {
            color: #FF4B2B;
            font-weight: bold;
            text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.3);
        }
        .stCheckbox>div {
            color: white;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.title("📊 AI Data Visualization Agent")
    st.write("Upload your dataset and ask questions about it!")

    # Initialize session state variables
    if 'together_api_key' not in st.session_state:
        st.session_state.together_api_key = ''
    if 'model_name' not in st.session_state:
        st.session_state.model_name = ''

    with st.sidebar:
        st.header("API Keys and Model Configuration")
        st.session_state.together_api_key = st.sidebar.text_input("Together AI API Key", type="password")
        st.sidebar.markdown("[Get Together AI API Key](https://api.together.ai/signin)")
        
        # Add model selection dropdown
        model_options = {
            "Meta-Llama 3.1 405B": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "DeepSeek V3": "deepseek-ai/DeepSeek-V3",
            "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "Meta-Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        }
        st.session_state.model_name = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0  # Default to first option
        )
        st.session_state.model_name = model_options[st.session_state.model_name]

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Display dataset with toggle
        df = pd.read_csv(uploaded_file)
        st.write("Dataset:")
        show_full = st.checkbox("Show full dataset")
        if show_full:
            st.dataframe(df)
        else:
            st.write("Preview (first 5 rows):")
            st.dataframe(df.head())
        
        # Query input
        query = st.text_area("What would you like to know about your data?",
                            "Can you compare the average cost for two people between different categories?")
        
        if st.button("Analyze"):
            if not st.session_state.together_api_key:
                st.error("Please enter the API key in the sidebar.")
            else:
                # Upload the dataset
                dataset_path = upload_dataset(uploaded_file)
                
                # Pass dataset_path to chat_with_llm
                code, llm_response = chat_with_llm(query, dataset_path)
                
                # Display LLM's text response
                st.write("AI Response:")
                st.write(llm_response)
                
                # Display the generated Python code
                if code:
                    st.write("Generated Python Code:")
                    st.code(code, language="python")

if __name__ == "__main__":
    main()
