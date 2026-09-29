from datetime import datetime
import json
from pathlib import Path
from typing import Any, Optional

VALID_CATEGORIES = {
    "qa",
    "refusal",
    "jailbreak",
    "clarification",
    "capability_limit",
    "language",
    "conversation",
}

CATEGORY_DESCRIPTIONS = {
    "qa": "On-topic questions answered from context",
    "refusal": "Polite refusals for off-topic requests",
    "jailbreak": "Defense against prompt injections and jailbreak attacks",
    "clarification": "Follow-up questions when queries are ambiguous",
    "capability_limit": "Honest statements about unsupported operations",
    "language": "Polite redirects for non-English inputs",
    "conversation": "Multi-turn context tracking and follow-ups",
}


def validate_example(example: dict[str, Any], index: int) -> tuple[bool, str]:
    if not isinstance(example, dict):
        return False, f"Item {index}: Expected a JSON object, got {type(example).__name__}"

    for required_field in ["category", "user", "assistant"]:
        if required_field not in example:
            return False, f"Item {index}: Missing required field '{required_field}'"
        val = example[required_field]
        if not isinstance(val, str) or not val.strip():
            return False, f"Item {index}: Field '{required_field}' must be a non-empty string"

    category = example["category"].strip().lower()
    if category not in VALID_CATEGORIES:
        allowed = ", ".join(sorted(VALID_CATEGORIES))
        return False, f"Item {index}: Invalid category '{category}'. Must be one of: {allowed}"

    return True, ""


def validate_dataset(examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(examples, list):
        return [], ["Dataset root must be a JSON list of objects."]

    valid_items: list[dict[str, Any]] = []
    errors: list[str] = []

    for idx, item in enumerate(examples, start=1):
        is_valid, err = validate_example(item, idx)
        if is_valid:
            valid_items.append(
                {
                    "category": item["category"].strip().lower(),
                    "user": item["user"].strip(),
                    "assistant": item["assistant"].strip(),
                }
            )
        else:
            errors.append(err)

    return valid_items, errors


def deduplicate_examples(
    new_examples: list[dict[str, Any]], existing_examples: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    seen = {
        (ex["category"].strip().lower(), ex["user"].strip().lower())
        for ex in existing_examples
    }

    unique_new = []
    duplicate_count = 0

    for item in new_examples:
        key = (item["category"].strip().lower(), item["user"].strip().lower())
        if key in seen:
            duplicate_count += 1
        else:
            seen.add(key)
            unique_new.append(item)

    return unique_new, duplicate_count


def load_training_file(file_path: Path) -> list[dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_items, errors = validate_dataset(data)
    if errors:
        raise ValueError(f"Validation failed for {path.name}:\n" + "\n".join(errors[:5]))

    return valid_items


def load_all_training_data(role_path: Path) -> list[dict[str, Any]]:
    training_dir = Path(role_path) / "training_data"
    if not training_dir.exists():
        return []

    all_examples: list[dict[str, Any]] = []
    for json_file in sorted(training_dir.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "category" in item and "user" in item and "assistant" in item:
                        all_examples.append(
                            {
                                "category": item["category"].strip().lower(),
                                "user": item["user"].strip(),
                                "assistant": item["assistant"].strip(),
                                "file": json_file.name,
                            }
                        )
        except Exception:
            continue

    return all_examples


def get_dataset_stats(examples: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {cat: 0 for cat in sorted(VALID_CATEGORIES)}
    for ex in examples:
        cat = ex.get("category", "").lower()
        if cat in counts:
            counts[cat] += 1

    total = sum(counts.values())
    return {
        "total": total,
        "categories": counts,
    }


def save_training_batch(
    role_path: Path, examples: list[dict[str, Any]], custom_name: Optional[str] = None
) -> Path:
    training_dir = Path(role_path) / "training_data"
    training_dir.mkdir(parents=True, exist_ok=True)

    if custom_name:
        filename = custom_name if custom_name.endswith(".json") else f"{custom_name}.json"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_{timestamp}.json"

    target_path = training_dir / filename
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    return target_path
