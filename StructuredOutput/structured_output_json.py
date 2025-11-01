from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash"
)

json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "description": "Name of the reviewer"
    },
    "email": {
      "type": ["string", "null"],
      "format": "email",
      "description": "Optional email address of the reviewer"
    },
    "summary": {
      "type": "string",
      "description": "Brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["Positive", "Negative", "Neutral"],
      "description": "Overall sentiment of the review"
    },
    "key_themes": {
      "type": "array",
      "description": "Key topics or recurring themes mentioned in the review",
      "items": {
        "type": "string"
      }
    },
    "pros": {
      "type": ["array", "null"],
      "description": "Positive aspects highlighted in the review",
      "items": {
        "type": "string"
      }
    },
    "cons": {
      "type": ["array", "null"],
      "description": "Negative aspects or drawbacks mentioned in the review",
      "items": {
        "type": "string"
      }
    }
  },
  "required": ["name", "summary", "sentiment", "key_themes"]
}

structured_model = model.with_structured_output(json_schema)

review = '''
The iPhone 15 marks a meaningful upgrade from previous standard models, bringing several Pro features 
into the base lineup while retaining Apple's premium build and ecosystem integration. Key enhancements such as the
 A16 Bionic chip, 48 MP main camera, USB-C port and the Dynamic Island make the experience feel more future-proof.
However, it still lacks some high-end features like a high refresh-rate display (120 Hz) and more advanced
zoom/telephoto optics, meaning power-users seeking top-tier specs might prefer stepping up to a Pro model.
Overall, it's a very compelling choice for those wanting a flagship experience without going all the way 
to the highest tier.
Pros: Top-tier performance from the A16 Bionic chip powering the device ensures smoothness and longevity. 
Major camera upgrade: The 48 MP main sensor (standard model) delivers significantly improved resolution and detail.
USB-C port adoption (a universal standard) improves compatibility and simplifies accessory/charger use. 
Cons: The display is still capped at 60 Hz refresh rate, which lags behind many competitors offering 90 or 120
Hz panels — meaning animations and scrolling feel less smooth. 
Charging speeds remain modest: Despite USB-C, wired charging remains around 20W and wireless charging hasn't seen 
major improvement.

Review by Abdullah Ahmed
email - abdullah.ahmed.pisces1010@gmail.com
'''

result = structured_model.invoke(review)
print(result)