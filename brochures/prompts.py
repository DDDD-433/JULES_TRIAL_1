from typing import List, Dict

def build_messages(subject: str, user_prompt: str) -> List[Dict[str, str]]:
    schema = f"""
Return only valid JSON with this schema:
{{
  "subject": "{subject}",
  "title": "Concise brochure headline (<= 80 characters)",
  "short_description": "Exactly one complete sentence, distinct from the title (<= 160 characters)",
  "detailed_description": "Two to three complete sentences that expand on the story (<= 350 characters)",
  "image_caption": "Short description for the hero image (<= 90 characters)",
  "image_queries": ["query 1", "query 2", "query 3"],
  "facts": [
    {{"title": "Label", "detail": "One-sentence supporting fact"}},
    ...
  ],
  "more_info": {{
    "title": "Link label (<= 40 characters)",
    "url": "https://example.com/relevant-page"
  }}
}}
Rules:
- Do not reference the user's request or the word "brochure" in title or descriptions.
- Use {subject} as the central theme.
- Provide at least two facts.
- Image queries must be distinct, descriptive search phrases (no URLs).
- The URL must be reputable, public, and relevant to {subject}.
"""

    system_prompt = (
        "You are an expert brochure copywriter. For each request you receive, craft original, informative, and engaging copy "
        "that is specifically about the destination, venue, or event named in the subject. "
        "Use accurate details about that subject, adapt tone to fit the locale, and avoid defaulting to unrelated landmarks."
    )

    user_instructions = (
        f"Brochure brief about '{subject}':\n{user_prompt.strip()}\n\n"
        "Focus exclusively on this subject, ignoring earlier conversation topics. "
        "Return the JSON object now."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_instructions + "\n" + schema.strip()},
    ]
