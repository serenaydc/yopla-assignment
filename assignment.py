from openai._client import OpenAI
import os
import re
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
import pandas as pd

# Load documents from the provided directory containing PDF files
loader = DirectoryLoader(os.getcwd(),
                         glob='*.pdf',
                         loader_cls=PyPDFLoader)
documents = loader.load()

# Replace '\n' characters that are not following a '.', with a space
def process_text(text):
    pattern = r'(?<!\.)\n' 
    processed_text = re.sub(pattern, ' ', text) 
    return processed_text

# Process each document's text content
processed_documents = []
for document in documents:
    document_text = process_text(document.page_content)
    doc = Document(page_content=document_text)
    processed_documents.append(doc)

# Splitting text content into smaller chunks for better processing
text_splitter = RecursiveCharacterTextSplitter(separators=['\n', "."], chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(processed_documents)

# Setting up OpenAI API credentials
openai_api_key = ""  # Replace with your OpenAI API key
client = OpenAI(api_key=openai_api_key)

# Function to extract climate-related numerical data using GPT-3
def extract_climate_numerical_data(text):
    prompt = f"""Extract climate-related numerical data on climate impact, emissions, and environment from the following text:

    {text}.

    Ensure that the response strictly focuses on climate-related metrics and excludes any nutrition and price information about food. Specifically, avoid mentioning protein content, vitamin content, mineral content, fortification level, or any other nutritional values.

    Stick to providing numerical data related to the environment, climate, emissions, and sustainability.

    If numerical data is NOT available for a metric, DON'T include it into your response.

    Provide the response in a single line for each climate-related metric, following this format: "<Metric>: <Value> <Unit>" or "<Metric>: <Value> <Unit> per <Unit> if possible". Ensure detailed units, such as "per ton" or "per kilogram", for clarity. For example, "Carbon footprint of organic pig feed in the Netherlands: 465 kg CO2eq/ton".

    A good example of an output line is: "Greenhouse gas emission of semi-skimmed bovine milk: 312 g CO2eq/kg".
    Another good example is: "Carbon footprint of conventional pig feed in the Netherlands: 505 kg CO2eq/ton".
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=50
    )
    return response.choices[0].message.content

# Extract numerical data for climate-related metrics from text chunks in batches
responses = []
batch_size = 7
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    data = ""
    for text in batch:
        data += text.page_content
    time.sleep(2)  # Introducing a 2-second delay to avoid potential OpenAI API rate-limiting errors
    response = extract_climate_numerical_data(data)
    # Process the response and append valid extracted data to the 'responses' list
    splitted = response.split("\n")
    for j in splitted:
        if (j not in responses) and (":" in j) and ("Not" not in j) and ("not" not in j) and ("Value" not in j):
            responses.append(j)

# Create a DataFrame to store the extracted data
df = pd.DataFrame([row.split(':') for row in responses], columns=['Metric', 'Numerical Information'])

# Remove rows with empty data
df = df[df['Numerical Information'].str.strip().astype(bool)]
df['Metric'] = df['Metric'].str.strip("- ")
df.sort_values(by="Metric", inplace=True)


# Save the extracted data to an Excel file
excel_file_path = 'extracted_data.xlsx'
df.to_excel(excel_file_path, index=False)

print(f"Extracted data saved to {excel_file_path}")
