from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table

from ..workflow.workflow import Workflow
from .evaluator import Evaluator

console = Console()


class TestSuite:

    def __init__(
        self, workflow: Workflow, dataset: List[Dict], evaluators: List[Evaluator]
    ):
        self.workflow = workflow
        self.dataset = dataset
        self.evaluators = evaluators

    async def arun(self) -> List[Dict]:
        results = []
        for i, initial_state in enumerate(self.dataset):
            console.print(f"\n--- Running Test Case {i + 1}/{len(self.dataset)} ---")
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
        table = Table(title="Test Suite Summary")
        table.add_column("Test Case", style="cyan")
        for evaluator in self.evaluators:
            table.add_column(evaluator.name, style="magenta")
        for i, result in enumerate(results):
            row = [f"Case #{i + 1}"]
            for eval_result in result["evaluations"]:
                row.append(str(eval_result.get("score", "N/A")))
            table.add_row(*row)
        console.print(table)
