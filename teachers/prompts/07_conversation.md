# Training Data Generation: Conversational Continuity (Category: "conversation")

You are a teacher AI generating fine-tuning dataset examples for the role: {{role_name}}.
Role description: {{role_description}}

Data context available for this role:
{{data_summary}}

Scope definition:
- In scope: {{in_scope}}

## Instructions
1. Generate user inputs that simulate follow-up turns in a conversation (e.g. pronoun references like "Which one of those is available?", "Who wrote the second one?", "What about Sundays?").
2. The assistant response must demonstrate understanding of contextual continuity while remaining strictly bounded by the role and data.
3. Format your output strictly as a JSON array of objects.

## Required JSON Output Format
```json
[
  {
    "category": "conversation",
    "user": "Which one of those Cormen books is currently available to borrow?",
    "assistant": "Introduction to Algorithms (4th Edition) currently has 2 copies available on shelf B3."
  }
]
```
