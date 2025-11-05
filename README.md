# ACE: Self-Improving Lab Extraction from Clinical Notes

An experiment inspired by Stanford’s Agentic Context Engineering (ACE) framework — applied to healthcare data abstraction.
This repo accompanies my Medium post
[The End of Prompt Engineering? Stanford’s Self-Improving AI Learned Clinical Reasoning on Its Own.](https://medium.com/p/6d8e0e1bc554)

It shows how a language model can reflect, learn, and build its own playbook for reasoning through complex clinical notes.
## What this repo includes

Core implementation of the ACE loop (extraction → reflection → curation)

Playbook evolution tracking — watch how reasoning strategies emerge over time

Notebook demo: ace_labs_demo.ipynb

→ Walks through 6 training notes, visualizes playbook evolution, and compares before/after results

Support for multiple models — works with Anthropic Claude or local LLM APIs

If you want to skip the setup and just see the results, check out the Medium article for context, visualizations, and analysis.

## Quickstart

Clone the repo and install dependencies:

```
pip install anthropic requests python-dotenv
```

Set your Anthropic API key in a .env file:

```
ANTHROPIC_API_KEY=your_api_key_here
```

Then open the notebook:

```
jupyter notebook ace_labs_demo.ipynb
```

You’ll see how the system starts from a blank prompt, critiques its own extractions, and improves with every patient note.

## Project Structure

```
ace_medium/
├── src/
│   ├── ace.py          # Main LabExtractionACE class
│   └── utils.py        # Utility functions for visualization
├── data/
│   └── synthetic_notes_journey.jsonl  # Synthetic clinical notes
├── ace_labs.ipynb      # Main notebook for lab extraction experiments
├── dev_ace.ipynb       # Development notebook with simpler ACE examples
└── README.md
```

## Background

Traditional prompt engineering relies on manual tweaks — rewriting instructions until the model performs better.
ACE changes that: it teaches models to improve by reflecting on their own errors and evolving their reasoning strategies.

This repo applies that idea to a classic hard problem in clinical NLP: lab value extraction — where models must reason through time, format chaos, and ambiguous references.

## Learn More

For the full write-up, results, and visualizations:
[Read the blog post on Medium](https://medium.com/p/6d8e0e1bc554)

For hands-on exploration:
[Run the demo notebook](https://github.com/mhdroz/ace_medium/blob/main/ace_labs.ipynb)

## Questions or feedback?

Let’s connect on [LinkedIn](https://www.linkedin.com/in/marie-humbert-droz/)

Interested in collaborating or working on a healthcare AI project? [Visit my website](https://www.mhd-ai.com/) to get in touch or book a call.

