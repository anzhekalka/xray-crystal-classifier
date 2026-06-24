"""
explainer.py sends the predicted crystal system to Mistral AI
and fetches a scientific explanation of its properties and significance.
"""
import requests
import os
from dotenv import load_dotenv 

load_dotenv() 
api_key = os.getenv("MISTRAL_API_KEY") 

def explain_prediction(crystal_system: str, confidence: float ) :
    """
    returns a scientific explanation of the predicted crystal system,
    provided by Mistral AI. Returns None if the request was not successful.
    """
    prompt = f"""You are a crystallographer explaining results to a student.

    An AI model analyzed an X-ray diffraction pattern and classified it as 
    {crystal_system.upper()} crystal system, with {confidence:.0%} confidence.

    Give exactly 3 bullet points explaining:
    1. What defines this crystal system geometrically (lattice angles and 
    side lengths)
    2. A real-world example material with this crystal system
    3. What the diffraction pattern of this crystal system typically looks 
    like and why (peak spacing, number of peaks, symmetry)

    Keep it scientifically accurate but accessible. No code names, no 
    extra sections, exactly 3 bullet points."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "mistral-small-latest",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 400
    }
    url = 'https://api.mistral.ai/v1/chat/completions'
    responce = requests.post(url, headers = headers, json = body)
    if responce.status_code == 200: 
        final_message = responce.json()["choices"][0]["message"]["content"] #json() conversion + extracting the content from a dictionary
        return final_message
    else:
        print ("API error:", responce.status_code, responce.text)
        return None 