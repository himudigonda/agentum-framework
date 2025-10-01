# agentum/testing.py
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table

from .agent import Agent
from .workflow import Workflow

console = Console()


class Evaluator:
    def __init__(self, name: str, evaluator_agent: Agent, instructions: str):
        self.name = name
        self.agent = evaluator_agent
        self.instructions = instructions

    async def evaluate(self, state: Dict[str, Any]) -> Dict:
        """Uses an agent to evaluate the final state of a workflow."""
        prompt = self.instructions.format(**state)
        response = await self.agent.llm.ainvoke(
            f"{self.agent.system_prompt}\n\n{prompt}"
        )
        # In a real scenario, you'd parse this response more robustly
        return {"evaluator": self.name, "score": response.content}


class TestSuite:
    def __init__(
        self, workflow: Workflow, dataset: List[Dict], evaluators: List[Evaluator]
    ):
        self.workflow = workflow
        self.dataset = dataset
        self.evaluators = evaluators

    async def arun(self) -> List[Dict]:
        """Runs the workflow against the dataset and evaluates each result."""
        results = []
        for i, initial_state in enumerate(self.dataset):
            console.print(f"\n--- Running Test Case {i+1}/{len(self.dataset)} ---")
            final_state = await self.workflow.arun(initial_state)

            evaluations = []
            for evaluator in self.evaluators:
                eval_result = await evaluator.evaluate(final_state)
                evaluations.append(eval_result)

            results.append(
                {
                    "initial_state": initial_state,
                    "final_state": final_state,
                    "evaluations": evaluations,
                }
            )
        return results

    def summary(self, results: List[Dict]):
        """Prints a summary table of the test results."""
        table = Table(title="Test Suite Summary")
        table.add_column("Test Case", style="cyan")
        # Add columns for your evaluators
        for evaluator in self.evaluators:
            table.add_column(evaluator.name, style="magenta")

        for i, result in enumerate(results):
            row = [f"Case #{i+1}"]
            for eval_result in result["evaluations"]:
                row.append(str(eval_result.get("score", "N/A")))
            table.add_row(*row)

        console.print(table)
