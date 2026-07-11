"""
Tracked-skills catalog.

This file is the only place domain knowledge lives for skill matching.
The matching engine (`_skills.py`) has no awareness of these keywords.
Replace with whatever skills are relevant to the roles you care about.

Deliberately a small, hand-picked, evidence-driven list, not an attempt
at an exhaustive taxonomy — see `docs/design-decisions.md`'s "Why
explicit extraction instead of skill ontology expansion?" for why this
is a design choice, not a limitation to "fix" by growing the list
speculatively. Every entry below was added because a real job posting
explicitly named that exact technology and the catalog didn't yet
track it — never added preemptively.
"""

TRACKED_SKILLS: list[str] = [
    "Python", "SQL", "AWS", "GCP", "Azure",
    "Docker", "Kubernetes", "PyTorch", "TensorFlow",
    "XGBoost", "MLflow", "ZenML", "Metaflow",
    "EC2", "EKS", "CloudFormation", "Cognito", "LLMs?",
]
