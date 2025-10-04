from typing import Any, Dict

from ..agent.agent import Agent


class Evaluator:

    def __init__(self, name: str, evaluator_agent: Agent, instructions: str):
        self.name = name
        self.agent = evaluator_agent
        self.instructions = instructions

    async def evaluate(self, state: Dict[str, Any]) -> Dict:
        prompt = self.instructions.format(**state)
        response = await self.agent.llm.ainvoke(
            f"{self.agent.system_prompt}\n\n{prompt}"
        )
        return {"evaluator": self.name, "score": response.content}
