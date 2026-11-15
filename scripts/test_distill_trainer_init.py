import inspect
import agenttune
from agenttune.agentic.events import EventLog, Event, EventKind

# Create dummy full-tier log
log = EventLog(tier="full")
log.append(Event(EventKind.OBSERVATION, {"text": "dummy task"}))
log.append(Event(EventKind.TEXT, {"text": "dummy response"}))
log.append(Event(EventKind.REWARD, {"value": 1.0}, scope="episode"))

class DummyFactory:
    def __call__(self, *, model, train_dataset, **kwargs):
        print(f"DummyFactory called with {len(train_dataset)} rows")
        class DummyTrainer:
            def train(self):
                print("DummyTrainer.train() executed successfully!")
                return "SUCCESS"
        return DummyTrainer()

trainer = agenttune.create_distill_trainer(
    student="dummy_model",
    trajectories=[log],
    trainer_factory=DummyFactory(),
    output_dir="./dummy_out",
    num_train_epochs=1
)

print(f"create_distill_trainer returned: {type(trainer)}")
res = trainer.train()
print(f"trainer.train() result: {res}")
