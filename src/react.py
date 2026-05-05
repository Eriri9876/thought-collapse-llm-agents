import re
from src.llm import chat
from src.tools import TOOLS

SYSTEM_PROMPTS = {
    "full": """You are a question-answering agent. Solve the question using search and calculation.

Use this format strictly:
Thought: reason about what to do next
Action: search[your query]  OR  calculate[math expression]  OR  finish[your final answer]

An Observation will be provided after each action.
Always write a Thought before every Action.
If searches are not yielding useful results, use finish[your best guess] rather than searching indefinitely.""",

    "none": """You are a question-answering agent. Solve the question using search and calculation.

Use this format strictly:
Action: search[your query]  OR  calculate[math expression]  OR  finish[your final answer]

An Observation will be provided after each action.
Do NOT write any Thought. Go directly to Action.
If searches are not yielding useful results, use finish[your best guess] rather than searching indefinitely.""",

    "compressed": """You are a question-answering agent. Solve the question using search and calculation.

Use this format strictly:
Thought: (10 words or fewer)
Action: search[your query]  OR  calculate[math expression]  OR  finish[your final answer]

An Observation will be provided after each action.
Keep every Thought to 10 words or fewer.
If searches are not yielding useful results, use finish[your best guess] rather than searching indefinitely.""",
}


def _parse_action(text: str) -> tuple[str, str] | None:
    match = re.search(r"Action:\s*(\w+)\[(.+?)\]", text, re.DOTALL)
    if not match:
        return None
    return match.group(1).lower(), match.group(2).strip()


def run_react(
    question: str,
    variant: str = "full",
    model: str = "deepseek-chat",
    max_steps: int = 8,
) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[variant]},
        {"role": "user", "content": f"Question: {question}"},
    ]
    trajectory = []

    for step in range(max_steps):
        response = chat(messages, model=model)
        messages.append({"role": "assistant", "content": response})

        parsed = _parse_action(response)
        if parsed is None:
            trajectory.append({"step": step, "response": response, "error": "parse_failed"})
            break

        action_type, action_input = parsed

        if action_type == "finish":
            trajectory.append({"step": step, "response": response, "action": "finish", "input": action_input})
            return {
                "answer": action_input,
                "trajectory": trajectory,
                "steps": step + 1,
                "status": "success",
            }

        observation = TOOLS.get(action_type, lambda q: f"Unknown action: {action_type}")(action_input)
        trajectory.append({
            "step": step,
            "response": response,
            "action": action_type,
            "input": action_input,
            "observation": observation,
        })
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return {
        "answer": None,
        "trajectory": trajectory,
        "steps": max_steps,
        "status": "max_steps_reached",
    }
