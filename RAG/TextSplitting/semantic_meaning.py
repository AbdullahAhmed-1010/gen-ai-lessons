from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

sample_text = '''
The city hums beneath a violet sky, lights blinking like tired eyes. Somewhere a screen flickers—code half-written,
words left unsaid. Memory tastes like static, sharp and almost sweet. The air moves heavy, carrying pieces of
unfinished music, broken algorithms of rain. A man runs without reason, chasing something that may have been
light once.
Dreams spill across pavement, pixel and dust, melting into motion. Someone laughs in another language.
The clock forgets to count.
Between the sound of traffic and thought, silence cracks open—only a breath wide—and something tries to
remember itself. Wires hum softly, echoing voices that built them. Electricity thinks it's alive.
Maybe it is, halfway. Maybe everything is halfway.
'''
docs = text_splitter.create_documents([sample_text])
print(len(docs))
print(docs)