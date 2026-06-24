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
    prompt = f"""You are a crystallographer explaining results to a curious 
student who is learning about X-ray diffraction for the first time.

An AI model analyzed an X-ray diffraction pattern and classified it as 
{crystal_system.upper()} crystal system, with {confidence:.0%} confidence.

Write a clear, engaging explanation covering:

1. **Geometric definition** — explain in 2-3 sentences what defines this 
   crystal system (lattice angles and side lengths), and why this 
   particular geometry exists in nature.

2. **Real-world examples** — name 2-3 real materials with this crystal 
   system, and briefly mention one interesting property or use of one 
   of them.

3. **Diffraction signature** — explain in detail what the diffraction 
   pattern of this crystal system typically looks like (peak spacing, 
   number of peaks, symmetry), and why the underlying atomic geometry 
   produces that specific pattern.

4. **Why this matters** — one or two sentences on why distinguishing 
   this crystal system matters in real research (materials science, 
   drug design, mineralogy, etc.)

Use clear formatted sections with headers. Be scientifically accurate, 
but write in an engaging, accessible style — like a good science 
communicator, not a dry textbook. Aim for around 200-300 words total.
Keep your complete response under 300 words so it fits comfortably 
within the response limit. Make sure to finish all sections completely 
— do not cut off mid-sentence."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "mistral-small-latest",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000
    }
    url = 'https://api.mistral.ai/v1/chat/completions'
    responce = requests.post(url, headers = headers, json = body, timeout=30)
    if responce.status_code == 200: 
        final_message = responce.json()["choices"][0]["message"]["content"] #json() conversion + extracting the content from a dictionary
        return final_message
    else:
        print ("API error:", responce.status_code, responce.text)
        return None 
    