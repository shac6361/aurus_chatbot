from pathlib import Path
from docling.document_converter import DocumentConverter
import chromadb
from chromadb.config import Settings
from nltk.tokenize import sent_tokenize
import nltk
from transformers import AutoTokenizer, AutoModel
import torch

# Ensure NLTK sentence tokenizer is available
nltk.download("punkt")
nltk.download("punkt_tab")

# Extract OCR text using Docling
image_path = Path("aurus\input\diagram1.png")
converter = DocumentConverter()
result = converter.convert(image_path)

# Aggregate non-empty OCR lines
orig_texts = [text.orig for text in result.document.texts if text.orig.strip()]
connected_text = " ".join(orig_texts)

# Chunk the text into sentences
chunks = sent_tokenize(connected_text)

# Load HuggingFace embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Define embedding function
def embed_text(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.cpu().numpy()

# Generate embeddings for chunks
embeddings = embed_text(chunks)

# Initialize Chroma client and collection
client = chromadb.Client(Settings())
collection = client.get_or_create_collection(name="ocr_documents")

# Insert chunks and embeddings into Chroma DB
collection.add(
    documents=chunks,
    ids=[f"doc_001_chunk_{i}" for i in range(len(chunks))],
    embeddings=embeddings,
    metadatas=[{
        "source": image_path.name,
        "chunk_index": i,
        "total_chunks": len(chunks)
    } for i in range(len(chunks))]
)

print(f"✅ Inserted {len(chunks)} chunks from {image_path.name} into Chroma DB.")

# Validate by retrieving one chunk
retrieved = collection.get(ids=["doc_001_chunk_0"])
print("\n📄 Sample Retrieved Chunk:")
print(retrieved["documents"][0])

# Writing to a file (overwriting existing content or creating a new file)
with open("aurus\output\ocr_output.txt", "w") as file_object:
    file_object.write(retrieved["documents"][0])

# Appending to a file
# with open("output.txt", "a") as file_object:
#     file_object.write("\nThis line is appended.")

########################################################################################
# from pathlib import Path
# from docling.document_converter import DocumentConverter
# import re
# from transformers import pipeline
# from textblob import TextBlob

# # Step 1: Extract OCR text using Docling
# converter = DocumentConverter()
# result = converter.convert(Path("swimlane_diagram.png"))
# # result = converter.convert(Path("aurus\input\swimlane_diagram.png"))

# # Get all non-empty OCR text lines
# orig_texts = [text.orig for text in result.document.texts if text.orig.strip()]
# connected_text = " ".join(orig_texts)

# # Optional: Print raw OCR output
# print("📝 Raw OCR Text:")
# print(connected_text)
#############################################################################3
# # Step 2: Spell correction using TextBlob
# blob = TextBlob(connected_text)
# corrected_text = str(blob.correct())

# print("\n🔧 Corrected Text:")
# print(corrected_text)

# # Step 3: Segment text by decision keywords
# segments = re.split(r'\b(NO|YES|Cancel the order|Finish|Deliver the order|Processing the payment)\b', corrected_text)
# steps = [seg.strip() for seg in segments if seg.strip()]

# # Optional: Print segmented steps
# print("\n🔍 Segmented Steps:")
# for step in steps:
#     print("-", step)

# Step 4: Summarize using LLM
# summarizer = pipeline("summarization", model="t5-small")

# # Join steps into a single string for LLM input
# summary_input = ". ".join(steps)

# summary = summarizer(summary_input, max_length=100, min_length=30, do_sample=False)

# # Final Output
# print("\n🧠 LLM Summary:")
# print(summary[0]['summary_text'])

###############################################################################