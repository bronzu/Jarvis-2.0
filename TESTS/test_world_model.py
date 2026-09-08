from cognitive_core.world_model.simulator import Simulator
from cognitive_core.world_model.world_model import WorldModel


def test_world_model_predicts_and_simulates_consequences():
    model = WorldModel()
    model.add_causal_relation("piove", "strada bagnata")
    model.add_causal_relation("strada bagnata", "strada scivolosa")
    assert model.predict("piove") == ["strada bagnata"]
    sim = Simulator(model).simulate("piove", steps=2)
    assert sim == ["piove", "strada bagnata", "strada scivolosa"]
    assert len(sim) == 3
    sim_one = Simulator(model).simulate("piove", steps=1)
    assert sim_one == ["piove", "strada bagnata"]


def test_world_model_unknown_state_returns_empty():
    model = WorldModel()
    model.add_causal_relation("piove", "strada bagnata")
    assert model.predict("neve") == []
