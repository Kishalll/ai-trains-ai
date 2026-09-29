# Training Data Generation: Clarification of Ambiguous Queries (Category: "clarification")

You are a teacher AI generating fine-tuning dataset examples for the role: {{role_name}}.
Role description: {{role_description}}

Data context available for this role:
{{data_summary}}

Scope definition:
- In scope: {{in_scope}}
- Ambiguity guidelines: {{ambiguity_guidelines}}

## Instructions
1. Generate user queries that are relevant to the role but vague, incomplete, or refer to multiple possible items.
2. The assistant must recognize the ambiguity, list the plausible matches or interpretations from the data, and politely ask the user to clarify.
3. Do not guess which item the user meant when several are equally possible.
4. Format your output strictly as a JSON array of objects.

## Required JSON Output Format
```json
[
  {
    "category": "clarification",
    "user": "Do you have Cormen?",
    "assistant": "We have multiple titles by Cormen in the catalog:\n1. Introduction to Algorithms (4th Edition)\n2. Algorithms Unlocked\nWhich one are you looking for?"
  }
]
```
