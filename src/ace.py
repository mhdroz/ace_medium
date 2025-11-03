import json
import anthropic
from typing import Dict
import requests
from enum import Enum

import anthropic


class LLMProvider(Enum):
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class LabExtractionACE:
    def __init__(self, provider, **kwargs):
        self.config = kwargs
        self.playbook = {
            "extraction_strategies": [],  # How to find labs in notes
            "validation_strategies": [],  # How to validate extracted labs
            "formatting_patterns": [],  # Common lab formatting patterns
        }
        self.history = []
        self.provider = provider
        self._setup_client()

    def _call_anthropic(self, system_prompt, user_prompt, temperature, stage="unknown"):
        """Call Anthropic API and track metrics"""
        import time

        print(f"  [{stage}] Temperature: {temperature}")

        start_time = time.time()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=8000,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        end_time = time.time()
        call_time = end_time - start_time

        print(
            f"    └─ Time: {call_time:.2f}s | Tokens: {response.usage.input_tokens + response.usage.output_tokens}"
        )

        return response.content[0].text

    def _call_local(self, system_prompt, user_prompt):
        """Call local LLM API"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        r = requests.post(self.api_url, json=payload, timeout=120)
        return r.json()["message"]["content"]

    def _setup_client(self):
        if self.provider == LLMProvider.ANTHROPIC:
            self.client = anthropic.Anthropic(api_key=self.config.get("api_key"))
            self.model = self.config.get("model", "claude-3-haiku-20240307")
            self.temperature = self.config.get("temperature", 0.1)
        elif self.provider == LLMProvider.LOCAL:
            self.api_url = self.config.get("api_url", "http://localhost:11434/api/chat")
            self.model = self.config.get("model", "gpt-oss:20b")
            self.temperature = 0.1

    def extract_labs(self, note_text: str, playbook_context: Dict) -> Dict:
        """
        STAGE 1: Extract lab values from clinical note
        """
        system_prompt = """You are a clinical lab extraction specialist. Extract lab values from clinical notes with high accuracy.

EXTRACTION GUIDELINES:
- Extract ALL lab values mentioned anywhere in the note
- For each lab, capture: name, value, unit, and date (if available)
- If multiple values for the same lab, extract ALL of them with their dates
- Common lab abbreviations: WBC (white blood cells), Hgb (hemoglobin), Plt (platelets), Na (sodium), K (potassium), Cr (creatinine), BUN (blood urea nitrogen)
- Watch for labs in: admission labs, daily labs, discharge labs, lab tables, narrative text
- Return ONLY valid JSON with no additional text or markdown formatting"""

        user_prompt = f"""PLAYBOOK (strategies learned from previous extractions):
{self._format_playbook(playbook_context, section='extraction')}

CLINICAL NOTE:
{note_text}

Extract all lab values mentioned in this note. Return JSON in this exact format:
{{
    "labs": [
        {{
            "name": "lab name",
            "value": "numeric value",
            "unit": "unit of measurement",
            "date": "date if mentioned (YYYY-MM-DD format, or 'not specified')",
            "context": "where found (e.g., 'admission labs', 'day 2', 'discharge labs')"
        }}
    ]
}}

Return ONLY the JSON, no other text."""

        if self.provider == LLMProvider.ANTHROPIC:
            result = self._call_anthropic(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.temperature,
                stage="extraction",
            )
        elif self.provider == LLMProvider.LOCAL:
            result = self._call_local(
                system_prompt=system_prompt, user_prompt=user_prompt
            )

        return self._parse_json_with_retry(
            result, system_prompt, user_prompt, "extraction"
        )

    def _parse_json_with_retry(
        self,
        result: str,
        system_prompt: str,
        user_prompt: str,
        stage: str,
        max_retries: int = 3,
    ) -> Dict:
        """
        Parse JSON result with retry mechanism for malformed JSON
        """
        for attempt in range(max_retries):
            try:
                # Clean up markdown formatting
                cleaned_result = result.strip()
                if cleaned_result.startswith("```json"):
                    cleaned_result = cleaned_result[7:]
                if cleaned_result.startswith("```"):
                    cleaned_result = cleaned_result[3:]
                if cleaned_result.endswith("```"):
                    cleaned_result = cleaned_result[:-3]

                # Try to parse JSON
                parsed = json.loads(cleaned_result.strip())
                return parsed

            except json.JSONDecodeError as e:
                print(
                    f"  ⚠ JSON parsing failed (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )

                if attempt < max_retries - 1:
                    # Retry with error feedback
                    print(f"  ↻ Retrying with error feedback...")

                    retry_prompt = f"""The previous response had a JSON formatting error: {str(e)}

    Previous response:
    {result}

    Please provide a corrected response. Remember:
    - Return ONLY valid JSON
    - No markdown formatting (no ```json or ```)
    - Ensure all quotes are properly escaped
    - Ensure all brackets and braces are properly closed
    - Ensure all property name enclosed in double quotes

    {user_prompt}"""

                    if self.provider == LLMProvider.ANTHROPIC:
                        result = self._call_anthropic(
                            system_prompt=system_prompt,
                            user_prompt=retry_prompt,
                            temperature=self.temperature,
                            stage=f"{stage}-retry-{attempt+1}",
                        )
                    elif self.provider == LLMProvider.LOCAL:
                        result = self._call_local(
                            system_prompt=system_prompt, user_prompt=retry_prompt
                        )
                else:
                    # Final attempt failed
                    print(f"  ✗ Failed to parse JSON after {max_retries} attempts")
                    print(f"  Raw response: {result}...")
                    raise

    def identify_most_recent(self, extraction: Dict, playbook_context: Dict) -> Dict:
        """
        STAGE 2: Identify the most recent value for each lab
        """
        system_prompt = """You are analyzing lab values to identify the most recent value for each unique lab test.

ANALYSIS GUIDELINES:
- Group labs by test name (e.g., all "sodium" values together)
- Identify the most recent value based on date/context
- Handle cases where dates are implicit (e.g., "admission labs" vs "discharge labs")
- If multiple values have same recency, note this ambiguity
- Return ONLY valid JSON with no additional text or markdown formatting"""

        user_prompt = f"""PLAYBOOK (validation strategies learned):
{self._format_playbook(playbook_context, section='validation')}

EXTRACTED LAB VALUES:
{json.dumps(extraction, indent=2)}

Identify the most recent value for each unique lab test. Return JSON in this format:
{{
    "most_recent_labs": [
        {{
            "name": "lab name",
            "value": "most recent value",
            "unit": "unit",
            "date": "date or context",
            "reasoning": "why this is the most recent"
        }}
    ],
    "ambiguous_cases": [
        {{
            "lab_name": "lab name",
            "issue": "description of ambiguity",
            "possible_values": ["value1", "value2"]
        }}
    ]
}}

Return ONLY the JSON, no other text."""

        if self.provider == LLMProvider.ANTHROPIC:
            result = self._call_anthropic(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.temperature,
                stage="identification",
            )
        elif self.provider == LLMProvider.LOCAL:
            result = self._call_local(
                system_prompt=system_prompt, user_prompt=user_prompt
            )

        return self._parse_json_with_retry(
            result, system_prompt, user_prompt, "identification"
        )

    def reflector(
        self,
        note_text: str,
        extraction: Dict,
        most_recent: Dict,
        ground_truth: Dict = None,
    ) -> Dict:
        """
        STAGE 3: Reflect on extraction quality
        """
        system_prompt = """You are reviewing lab extraction quality. Analyze what was done well and what could be improved.

REVIEW GUIDELINES:
- Check if all labs mentioned in the note were captured
- Verify correct identification of most recent values
- Look for missed labs in tables, narrative text, or headers
- Identify patterns that would improve future extractions
- If ground truth is provided, compare against it
- Return ONLY valid JSON with no additional text or markdown formatting"""

        # Create abbreviated note excerpt for context
        note_excerpt = note_text[:1500] + "..." if len(note_text) > 1500 else note_text

        user_prompt = f"""ORIGINAL NOTE (excerpt):
{note_excerpt}

EXTRACTION RESULTS:
{json.dumps(extraction, indent=2)}

MOST RECENT LAB IDENTIFICATION:
{json.dumps(most_recent, indent=2)}"""

        if ground_truth:
            user_prompt += f"""

GROUND TRUTH (expected labs):
{json.dumps(ground_truth, indent=2)}"""

        user_prompt += """

Analyze the extraction quality. Return JSON in this format:
{
    "extraction_quality": {
        "labs_found": ["list of labs successfully extracted"],
        "labs_missed": ["list of labs that should have been extracted but weren't"],
        "incorrect_extractions": ["list of any incorrect extractions"]
    },
    "most_recent_identification_quality": {
        "correct_identifications": ["labs where most recent value was correctly identified"],
        "incorrect_identifications": ["labs where wrong value was chosen as most recent"]
    },
    "learned_patterns": ["actionable strategies for future extractions"]
}

Return ONLY the JSON, no other text."""

        if self.provider == LLMProvider.ANTHROPIC:
            result = self._call_anthropic(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                stage="reflection",
            )
        elif self.provider == LLMProvider.LOCAL:
            result = self._call_local(
                system_prompt=system_prompt, user_prompt=user_prompt
            )

        return self._parse_json_with_retry(
            result, system_prompt, user_prompt, "reflection"
        )

    def curator(self, reflection: Dict) -> Dict:
        """
        STAGE 4: Update playbook based on reflections
        """
        print("\n[CURATOR] Updating playbook...")

        # Add learned patterns to appropriate strategies
        for pattern in reflection.get("learned_patterns", []):
            # Categorize patterns
            if any(
                keyword in pattern.lower()
                for keyword in [
                    "extract",
                    "look for",
                    "check",
                    "scan",
                    "find",
                    "capture",
                ]
            ):
                if pattern not in self.playbook["extraction_strategies"]:
                    self.playbook["extraction_strategies"].append(pattern)
                    print(f"  ✓ Added extraction strategy: {pattern[:80]}...")

            elif any(
                keyword in pattern.lower()
                for keyword in [
                    "validate",
                    "verify",
                    "most recent",
                    "latest",
                    "compare",
                    "date",
                ]
            ):
                if pattern not in self.playbook["validation_strategies"]:
                    self.playbook["validation_strategies"].append(pattern)
                    print(f"  ✓ Added validation strategy: {pattern[:80]}...")

            elif any(
                keyword in pattern.lower()
                for keyword in ["format", "pattern", "structure", "table", "section"]
            ):
                if pattern not in self.playbook["formatting_patterns"]:
                    self.playbook["formatting_patterns"].append(pattern)
                    print(f"  ✓ Added formatting pattern: {pattern[:80]}...")

        # Keep playbook manageable (top 10 items per category)
        self.playbook["extraction_strategies"] = self.playbook["extraction_strategies"][
            -10:
        ]
        self.playbook["validation_strategies"] = self.playbook["validation_strategies"][
            -10:
        ]
        self.playbook["formatting_patterns"] = self.playbook["formatting_patterns"][
            -10:
        ]

        print(f"\nPlaybook now contains:")
        print(
            f"  - {len(self.playbook['extraction_strategies'])} extraction strategies"
        )
        print(
            f"  - {len(self.playbook['validation_strategies'])} validation strategies"
        )
        print(f"  - {len(self.playbook['formatting_patterns'])} formatting patterns")

        return self.playbook

    def _format_playbook(self, playbook: Dict, section: str = "all") -> str:
        """Format playbook for prompt inclusion"""
        if not any(playbook.values()):
            return "No strategies learned yet. This is your first case."

        formatted = ""

        if section in ["all", "extraction"] and playbook.get("extraction_strategies"):
            formatted += "EXTRACTION STRATEGIES:\n"
            for i, strategy in enumerate(playbook["extraction_strategies"], 1):
                formatted += f"{i}. {strategy}\n"

        if section in ["all", "validation"] and playbook.get("validation_strategies"):
            if formatted:
                formatted += "\n"
            formatted += "VALIDATION STRATEGIES:\n"
            for i, strategy in enumerate(playbook["validation_strategies"], 1):
                formatted += f"{i}. {strategy}\n"

        if section in ["all", "formatting"] and playbook.get("formatting_patterns"):
            if formatted:
                formatted += "\n"
            formatted += "FORMATTING PATTERNS:\n"
            for i, pattern in enumerate(playbook["formatting_patterns"], 1):
                formatted += f"{i}. {pattern}\n"

        return formatted if formatted else "No strategies learned yet for this section."

    def process_note(
        self, note_id: int, note_text: str, ground_truth: Dict = None
    ) -> tuple:
        """
        Process a single note through the ACE workflow
        """
        print(f"\n{'='*60}")
        print(f"PROCESSING NOTE {note_id}")
        print(f"{'='*60}")

        # STAGE 1: Extract labs
        print("\n[STAGE 1: EXTRACTION]")
        extraction = self.extract_labs(note_text, self.playbook)
        lab_count = len(extraction.get("labs", []))
        print(f"  ✓ Extracted {lab_count} lab values")

        # STAGE 2: Identify most recent
        print("\n[STAGE 2: IDENTIFY MOST RECENT]")
        most_recent = self.identify_most_recent(extraction, self.playbook)
        unique_labs = len(most_recent.get("most_recent_labs", []))
        ambiguous = len(most_recent.get("ambiguous_cases", []))
        print(f"  ✓ Identified {unique_labs} unique labs")
        if ambiguous > 0:
            print(f"  ⚠ {ambiguous} ambiguous cases flagged")

        # STAGE 3: Reflect
        print("\n[STAGE 3: REFLECTION]")
        reflection = self.reflector(note_text, extraction, most_recent, ground_truth)
        pattern_count = len(reflection.get("learned_patterns", []))
        print(f"  ✓ Learned {pattern_count} new patterns")

        if ground_truth:
            missed = len(
                reflection.get("extraction_quality", {}).get("labs_missed", [])
            )
            if missed > 0:
                print(f"  ⚠ Missed {missed} labs compared to ground truth")

        # STAGE 4: Update playbook
        print("\n[STAGE 4: CURATION]")
        self.playbook = self.curator(reflection)

        # Store history
        self.history.append(
            {
                "note_id": note_id,
                "extraction": extraction,
                "most_recent": most_recent,
                "reflection": reflection,
                "playbook_snapshot": self.playbook.copy(),
            }
        )

        return extraction, most_recent, reflection

    def show_playbook_evolution(self):
        """Show how the playbook evolved over time"""
        print(f"\n{'='*60}")
        print("PLAYBOOK EVOLUTION")
        print(f"{'='*60}")

        for i, history_item in enumerate(self.history, 1):
            pb = history_item["playbook_snapshot"]
            print(f"\nAfter Note {history_item['note_id']}:")
            print(
                f"  Extraction strategies: {len(pb.get('extraction_strategies', []))}"
            )
            print(
                f"  Validation strategies: {len(pb.get('validation_strategies', []))}"
            )
            print(f"  Formatting patterns: {len(pb.get('formatting_patterns', []))}")

        print(f"\n{'='*60}")
        print("FINAL PLAYBOOK")
        print(f"{'='*60}")
        print(self._format_playbook(self.playbook))

    def compare_with_without_playbook(
        self, note_text: str, ground_truth: Dict = None
    ) -> Dict:
        """
        Compare extraction with empty vs learned playbook
        """
        print(f"\n{'='*60}")
        print(f"COMPARISON: Empty Playbook vs Learned Playbook")
        print(f"{'='*60}")

        # Save current playbook
        current_playbook = self.playbook.copy()

        # ===== RUN 1: Empty Playbook =====
        print("\n[RUN 1: Empty Playbook]")
        empty_playbook = {
            "extraction_strategies": [],
            "validation_strategies": [],
            "formatting_patterns": [],
        }

        extraction_empty = self.extract_labs(note_text, empty_playbook)
        most_recent_empty = self.identify_most_recent(extraction_empty, empty_playbook)

        # ===== RUN 2: Learned Playbook =====
        print("\n[RUN 2: Learned Playbook]")
        extraction_learned = self.extract_labs(note_text, current_playbook)
        most_recent_learned = self.identify_most_recent(
            extraction_learned, current_playbook
        )

        # ===== COMPARE =====
        labs_empty = {
            lab["name"].lower(): lab
            for lab in most_recent_empty.get("most_recent_labs", [])
        }
        labs_learned = {
            lab["name"].lower(): lab
            for lab in most_recent_learned.get("most_recent_labs", [])
        }

        newly_found = [name for name in labs_learned if name not in labs_empty]

        print(f"\n{'='*60}")
        print("COMPARISON RESULTS")
        print(f"{'='*60}")
        print(f"\nEmpty Playbook:")
        print(f"  Labs found: {len(labs_empty)}")

        print(f"\nLearned Playbook:")
        print(f"  Labs found: {len(labs_learned)}")
        print(f"  Additional labs found: {len(newly_found)}")

        if newly_found:
            print(f"\n  New labs caught:")
            for lab_name in newly_found:
                lab = labs_learned[lab_name]
                print(f"    • {lab['name']}: {lab['value']} {lab['unit']}")

        # Compare against ground truth if provided
        if ground_truth:
            print(f"\n{'='*60}")
            print("GROUND TRUTH COMPARISON")
            print(f"{'='*60}")

            gt_labs = {
                lab["name"].lower(): lab
                for lab in ground_truth.get("most_recent_labs", [])
            }

            empty_recall = (
                len(set(labs_empty.keys()) & set(gt_labs.keys())) / len(gt_labs)
                if gt_labs
                else 0
            )
            learned_recall = (
                len(set(labs_learned.keys()) & set(gt_labs.keys())) / len(gt_labs)
                if gt_labs
                else 0
            )

            print(f"\nEmpty Playbook Recall: {empty_recall:.1%}")
            print(f"Learned Playbook Recall: {learned_recall:.1%}")
            print(f"Improvement: {(learned_recall - empty_recall):.1%}")

        return {
            "empty_playbook": {
                "extraction": extraction_empty,
                "most_recent": most_recent_empty,
                "labs_found": len(labs_empty),
            },
            "learned_playbook": {
                "extraction": extraction_learned,
                "most_recent": most_recent_learned,
                "labs_found": len(labs_learned),
            },
            "newly_found_labs": newly_found,
            "improvement": len(newly_found),
        }
