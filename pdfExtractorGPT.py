from autogen.agent import Agent
from autogen.io import FileInput, TextOutput
from autogen.process import FunctionProcess
import openai
import PyPDF2
import re

# Set your OpenAI API key
openai.api_key = 'your-api-key'

# Agent 1: Convert PDF to text using GPT API
def pdf_to_text(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ''
        for page_num in range(reader.numPages):
            page_text = reader.getPage(page_num).extractText()
            text += page_text
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=text,
        max_tokens=2048,
        temperature=0,
        stop=None
    )
    return response.choices[0].text.strip()

agent1 = Agent(
    name="PDF to Text Agent",
    process=FunctionProcess(pdf_to_text),
    inputs=FileInput(),
    outputs=TextOutput()
)

# Agent 2: Extract Table of Contents using GPT API
def extract_table_of_contents(text):
    toc = []
    lines = text.split('\n')
    for line in lines:
        if re.match(r'^\d+\.\s+.+', line):
            toc.append(line.strip())
    return toc

agent2 = Agent(
    name="Table of Contents Agent",
    process=FunctionProcess(extract_table_of_contents),
    inputs=TextOutput(),
    outputs=TextOutput()
)

# Agent 3: Embed and Vectorize using GPT API
def embed_and_vectorize(text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=text,
        max_tokens=2048,
        temperature=0,
        stop=None
    )
    return response.choices[0].text.strip()

agent3 = Agent(
    name="Embed and Vectorize Agent",
    process=FunctionProcess(embed_and_vectorize),
    inputs=TextOutput(),
    outputs=TextOutput()
)

# Connect agents
agent1.connect(agent2)
agent2.connect(agent3)

# Run agents
pdf_path = "Introduction-to-Management.pdf"  # Provide the path to the PDF file
output_text = agent1.run(pdf_path)

# Output from agent 1 is already the input for agent 2
toc = agent2.run(output_text)

# Output from agent 2 is the input for agent 3
embeddings = agent3.run(toc)

print("Embeddings generated:", embeddings)
