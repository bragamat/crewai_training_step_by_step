from crewai.events import (
    BaseEventListener,
    LiteAgentExecutionCompletedEvent,
    LiteAgentExecutionStartedEvent,
    TaskCompletedEvent,
)
from crewai.task import Task
from crewai.utilities.evaluators.task_evaluator import TaskEvaluator
from opik.evaluation.metrics import Hallucination


class EvalListenerSetup(BaseEventListener):
    def __init__(self):
        super().__init__()
        # Store evaluation results directly as instance attributes
        self.hallucination_score = None
        self.hallucination_reason = None

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(TaskCompletedEvent)
        def on_task_completed(source: Task, event: TaskCompletedEvent):
            if source.name == "reporting_task":
                evaluator = TaskEvaluator(source.agent)
                evaluation = evaluator.evaluate(source)

                hallucination_eval = Hallucination()
                hallucination_result = hallucination_eval.score(
                    input=source.prompt() + "\n\n" + source.prompt_context,
                    output=event.output,
                )
                print("evaluation", evaluation.quality)
                print("evaluation", evaluation.suggestions)
                print("------")
                print(hallucination_result.value)
                print(hallucination_result.reason)

                # Store evaluation results directly on this instance
                self.hallucination_score = hallucination_result.value
                self.hallucination_reason = hallucination_result.reason
