# Text Summarization App

This is a simple text summarization application built with Streamlit and TensorFlow. It allows users to enter text and generate summaries using different models like T5, PEGASUS, and BART.

## Getting Started

These instructions will help you run the project on your local machine.

### Prerequisites

You will need the following libraries installed:

- Streamlit
- Transformers
- TensorFlow  
- OpenAI API (for GPT-3)

You can install them by running:

pip install streamlit transformers tensorflow openai


You will also need to sign up for an [OpenAI API key](https://openai.com/api/) to use the GPT-3 model.

### Installing

Clone the repository:

git clone https://github.com/YawningFold/Text-Summarization-Using-Transformers-Implementation-.git

Navigate to the project directory:

cd Text-Summarization-Using-Transformers-Implementation-

### Running the app

Execute:

streamlit run app.py
This will start the Streamlit app on http://localhost:8501


This will start the Streamlit app on `http://localhost:8501`

## Built With

- [Streamlit](https://streamlit.io/) - For creating the web app
- [TensorFlow](https://www.tensorflow.org/) - For loading pretrainined models like T5, PEGASUS, BART  
- [Transformers](https://huggingface.co/transformers/) - For tokenization
- [OpenAI API](https://openai.com/api/) - For accessing GPT-3
