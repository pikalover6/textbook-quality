import json
import asyncio
import httpx
from tqdm import tqdm
import difflib

# Constants & Global Variables
API_URL = "https://api.together.xyz/inference"
API_KEY = ""
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}
TIMEOUT = 30
SEMAPHORE = asyncio.Semaphore(25)

GENERAL_TOPICS = {
    'world history': ['Ancient Civilizations', 'Global Revolutions', 'Renaissance', 'World Wars', 'Decolonization and Globalization'],
    'chemistry': ['Atomic Structure', 'Chemical Bonds', 'Biochemistry', 'Chemical Thermodynamics', 'Environmental Chemistry'],
    'philosophy': ['Metaphysics', 'Ethics', 'Existentialism', 'Philosophy of Science', 'Eastern Philosophy'],
    'mathematics': ['Algebra', 'Geometry', 'Calculus', 'Differential Equations', 'Mathematical Logic'],
    'biology': ['Cell Biology', 'Evolution', 'Genetics', 'Ecology', 'Human Physiology'],
    'physics': ['Mechanics', 'Electromagnetism', 'Quantum Physics', 'Thermal Physics', 'Astrophysics'],
    'arts and creativity': ['Art']
}
GRADE_LEVELS = ['7th grade', 'high school', 'college']

async def make_api_request(payload):
    """Utility function to make API requests and handle potential errors."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(API_URL, json=payload, headers=HEADERS)
        response.raise_for_status()  # Raises an HTTPError if an error occurred.
        return response.json()['output']['choices'][0]['text'].strip()

ALL_TOPICS = []  # Global list to store all generated topics.

def is_similar(a, b, threshold=0.9):
    """Check if two strings are similar based on the given threshold."""
    similarity = difflib.SequenceMatcher(None, a, b).ratio()
    return similarity >= threshold

MAX_RETRIES = 5  # Define a constant for maximum retries.

async def generate_topic_list_async(base_topic, grade_level, num_topics=5, pbar=None):
    prompt = f"<human>: Generate {num_topics} unique topics related to '{base_topic}' suitable for a {grade_level} textbook, without explaining them.\n<bot>:"
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 2048,
        "temperature": 0.7
    }

    retries = 0
    unique_topics = []
    while len(unique_topics) < num_topics and retries < MAX_RETRIES:
        topics_text = await make_api_request(payload)
        
        if pbar:  # Increment the progress bar for topic generation.
            pbar.update(1) 

        # Filter out lines that don't start with a number.
        topics = [topic.strip().split('. ')[-1] for topic in topics_text.split('\n') if topic and topic[0].isdigit()]

        # De-duplication logic with respect to all topics generated so far
        for topic in topics:
            if not any(is_similar(topic, existing_topic) for existing_topic in ALL_TOPICS):  # Check against ALL_TOPICS
                unique_topics.append(topic)
                ALL_TOPICS.append(topic)  # Update ALL_TOPICS with the new topic

        retries += 1

    if len(unique_topics) < num_topics:
        print(f"Warning: Could not generate {num_topics} unique topics for {base_topic} after {MAX_RETRIES} attempts. Only generated {len(unique_topics)} topics.")
    
    return unique_topics

async def generate_entry_async(topic, grade_level, pbar=None):
    async with SEMAPHORE:
        prompt = f"<human>: Explain the topic '{topic}' in detail suitable for a {grade_level} textbook.\n<bot>:"
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "max_tokens": 2048,
            "temperature": 0.7
        }        
        entry = await make_api_request(payload)
        
        # Remove the first line if it starts with "Sure".
        entry_lines = entry.split('\n')
        if entry_lines[0].strip().startswith("Sure"):
            entry = '\n'.join(entry_lines[1:])

        entry = entry.strip()
        
        if pbar:
            pbar.update(1)
        return topic, entry

async def generate_textbook_for_grade(grade_level):
    textbook = {}
    
    with tqdm(total=len(GENERAL_TOPICS), desc=f"Generating topics for {grade_level}") as pbar_topics:  # Progress bar for topics.
        all_topics_for_grade = []
        
        for base_topic in GENERAL_TOPICS.keys():
            topics_for_base_topic = await generate_topic_list_async(base_topic, grade_level, num_topics=5, pbar=pbar_topics)
            all_topics_for_grade.extend(topics_for_base_topic)

        with tqdm(total=len(all_topics_for_grade), desc=f"Generating entries for {grade_level}") as pbar_entries:  # Progress bar for entries.
            entries_for_grade = await asyncio.gather(*[generate_entry_async(topic, grade_level, pbar=pbar_entries) for topic in all_topics_for_grade])

    for topic, content in entries_for_grade:
        textbook[topic] = content
        
    return textbook

async def generate_textbooks_async():
    all_textbooks = {}
    
    for grade_level in tqdm(GRADE_LEVELS, desc="Generating textbooks for grade levels"):
        textbook_for_grade = await generate_textbook_for_grade(grade_level)
        all_textbooks[f"{grade_level} Textbook"] = textbook_for_grade

    with open("all_textbooks.json", "w") as file:
        json.dump(all_textbooks, file, indent=4)

# Run the asynchronous function
if __name__ == "__main__":
    try:
        asyncio.run(generate_textbooks_async())
    except Exception as e:
        print(f"An error occurred: {e}")
