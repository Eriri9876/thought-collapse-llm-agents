import re
from src.llm import chat
from src.tools import TOOLS
from src import cost_tracker

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
    max_tokens_per_call: int | None = None,
    max_total_output_tokens: int | None = None,
) -> dict:
    """Run a ReAct trajectory.

    ``max_tokens_per_call`` caps each LLM call's completion tokens (``None``
    leaves it provider-default — the legacy behaviour for Qwen/V3 runs).
    ``max_total_output_tokens`` caps the cumulative completion tokens across
    the entire trajectory; when set we reset and read the per-trace
    counter in ``cost_tracker`` before each turn. Hitting it returns status
    ``output_budget_exceeded``. These caps were added for cross-family
    Llama-3.1 runs where a small model can loop and burn the budget; they
    are no-ops when left at ``None``.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[variant]},
        {"role": "user", "content": f"Question: {question}"},
    ]
    trajectory = []

    if max_total_output_tokens is not None:
        cost_tracker.reset_run_counter()

    for step in range(max_steps):
        response = chat(messages, model=model, max_tokens=max_tokens_per_call)
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

        if (
            max_total_output_tokens is not None
            and cost_tracker.get_run_completion_tokens() >= max_total_output_tokens
        ):
            return {
                "answer": None,
                "trajectory": trajectory,
                "steps": step + 1,
                "status": "output_budget_exceeded",
            }

    return {
        "answer": None,
        "trajectory": trajectory,
        "steps": max_steps,
        "status": "max_steps_reached",
    }
