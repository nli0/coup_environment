from env import raw_env
from pettingzoo.utils import agent_selector, wrappers
from utils import ActionSpecification, Action, AgentStatus, Cards


def test_init_env():
    env = raw_env(render_mode="human", player_count=2)
    env.reset(seed=42)

    # start with player 0
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.INCOME)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    print("hi")


def test_duke_challenge_succeed():
    env = raw_env(render_mode="human", player_count=2)
    env.reset(seed=42)

    env.players[0].hidden_cards = [Cards.ASSASSIN, Cards.AMBASSADOR]
    env.players[1].hidden_cards = [Cards.DUKE, Cards.DUKE]

    # start with player 0
    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.DUKE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    # player 1 can challenge
    assert env.agent_selection == "1"
    assert env.status == AgentStatus.CAN_CHALLENGE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    # challenge succeeds, so player 0 loses an influence
    assert env.agent_selection == "0"
    assert env.status == AgentStatus.LOSE_INFLUENCE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(
        Action.LOSE_INFLUENCE, target=1
    )  # lose the Ambassador
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2
    assert len(agent.hidden_cards) == 1
    assert agent.hidden_cards[0] == Cards.ASSASSIN
    assert len(agent.visible_cards) == 1
    assert agent.visible_cards[0] == Cards.AMBASSADOR
    print("hi")


def test_assassination_fails_contessa_bluff():
    env = raw_env(render_mode="human", player_count=2)
    env.reset(seed=42)

    env.players[0].hidden_cards = [Cards.DUKE, Cards.AMBASSADOR]
    env.players[1].hidden_cards = [Cards.CONTESSA, Cards.DUKE]

    # start with player 0
    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.DUKE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    # player 1 can challenge
    assert env.agent_selection == "1"
    assert env.status == AgentStatus.CAN_CHALLENGE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    # challenge FAILS, so player 1 loses an influence
    assert env.agent_selection == "1"
    assert env.players[0].coins == 5
    assert env.players[1].coins == 2
    assert env.status == AgentStatus.LOSE_INFLUENCE
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(
        Action.LOSE_INFLUENCE, target=0
    )  # lose the Contessa
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 5
    assert env.players[1].coins == 2
    assert len(agent.hidden_cards) == 1
    assert agent.hidden_cards[0] == Cards.DUKE
    assert len(agent.visible_cards) == 1
    assert agent.visible_cards[0] == Cards.CONTESSA

    # player 1 takes income
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 5
    assert env.players[1].coins == 2
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.INCOME)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.players[0].coins == 5
    assert env.players[1].coins == 3
    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN

    # player 0 dukes
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.DUKE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.status == AgentStatus.CAN_CHALLENGE
    assert env.players[0].coins == 5
    assert env.players[1].coins == 3

    # player 1 does not challenge
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.NO_CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 8
    assert env.players[1].coins == 3

    # player 1 takes income again
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.INCOME)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.players[0].coins == 8
    assert env.players[1].coins == 4
    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN

    # player 0 coups player 1
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.COUP, target=1)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.players[0].coins == 1
    assert env.players[1].coins == 4

    # player 1 chooses which card to lose
    assert env.agent_selection == "1"
    assert env.status == AgentStatus.LOSE_INFLUENCE
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.LOSE_INFLUENCE, target=0)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.done == True
    assert env.players[0].coins == 1
    assert env.players[1].coins == 4
    assert env.players[1].hidden_cards == []
    assert len(env.players[1].visible_cards) == 2


def test_assassination_succeeds_contessa_bluff():
    # 0 assassinates 1, 1 doesn't challenge. 1 tries to block the assassination (no contessa), 0 challenges, 1 loses both cards

    env = raw_env(render_mode="human", player_count=2)
    env.reset(seed=42)

    env.players[0].hidden_cards = [Cards.ASSASSIN, Cards.AMBASSADOR]
    env.players[1].hidden_cards = [Cards.DUKE, Cards.CAPTAIN]

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # start with player 0 income
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.INCOME)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 3
    assert env.players[1].coins == 2

    # player 1 income
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.INCOME)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 3
    assert env.players[1].coins == 3

    # player 0 assassinates player 1
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.ASSASSIN, target=1)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.players[0].coins == 0
    assert env.players[1].coins == 3
    assert env.status == AgentStatus.CAN_CHALLENGE

    # player 1 does not challenge
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.NO_CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.players[0].coins == 0
    assert env.players[1].coins == 3
    assert env.status == AgentStatus.ASSASSINATED

    # player 1 tries to block assassination

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.BLOCK_ASSASSINATION)  # lose the Contessa
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.players[0].coins == 0
    assert env.players[1].coins == 3
    assert env.status == AgentStatus.CAN_CHALLENGE

    # player 0 challenges; player 1 LOSES BOTH CARDS

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CHALLENGE)  # lose the Contessa
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.players[0].coins == 0
    assert env.players[1].coins == 3
    assert len(env.players[0].hidden_cards) == 0
    assert len(env.players[0].visible_cards) == 2


def test_assassination_challenge_fails_has_contessa():
    # 0 assassinates 1, 1 challenges. 0 actually has assassin, 1 loses a card (1 has a contessa)

    env = raw_env(render_mode="human", player_count=2)
    env.reset(seed=42)

    env.players[0].hidden_cards = [Cards.ASSASSIN, Cards.AMBASSADOR]
    env.players[1].hidden_cards = [Cards.CONTESSA, Cards.CAPTAIN]

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # start with player 0 income
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.INCOME)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 3
    assert env.players[1].coins == 2

    # player 1 income
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.INCOME)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 3
    assert env.players[1].coins == 3

    # player 0 assassinates player 1
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.ASSASSIN, target=1)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.players[0].coins == 0
    assert env.players[1].coins == 3
    assert env.status == AgentStatus.CAN_CHALLENGE

    # player 1 challenges
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.players[0].coins == 0
    assert env.players[1].coins == 3
    assert env.status == AgentStatus.LOSE_INFLUENCE

    # player 1 fails, lose influence (the contessa)
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.LOSE_INFLUENCE, 0)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.players[0].coins == 0
    assert env.players[1].coins == 3
    assert env.players[1].hidden_cards == [Cards.CAPTAIN]
    assert env.players[1].visible_cards == [Cards.CONTESSA]


def test_assassination_challenge_fails_no_contessa():
    # 0 assassinates 1, 1 challenges. 0 actually has assassin, 1 loses all cards (1 has no contessa)

    env = raw_env(render_mode="human", player_count=2)
    env.reset(seed=42)

    env.players[0].hidden_cards = [Cards.ASSASSIN, Cards.AMBASSADOR]
    env.players[1].hidden_cards = [Cards.AMBASSADOR, Cards.CAPTAIN]

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # start with player 0 income
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.INCOME)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 3
    assert env.players[1].coins == 2

    # player 1 income
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.INCOME)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 3
    assert env.players[1].coins == 3

    # player 0 assassinates player 1
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.ASSASSIN, target=1)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.players[0].coins == 0
    assert env.players[1].coins == 3
    assert env.status == AgentStatus.CAN_CHALLENGE

    # player 1 challenges
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.players[0].coins == 0
    assert env.players[1].coins == 3

    assert len(env.players[1].hidden_cards) == 0
    assert len(env.players[1].visible_cards) == 2


def test_assassination_succeeds_no_challenge():
    # 0 assassinates 1, 1 doesn't challenge; 1 loses one influence

    env = raw_env(render_mode="human", player_count=2)
    env.reset(seed=42)

    env.players[0].hidden_cards = [Cards.ASSASSIN, Cards.AMBASSADOR]
    env.players[1].hidden_cards = [Cards.AMBASSADOR, Cards.CAPTAIN]

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # start with player 0 income
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.INCOME)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 3
    assert env.players[1].coins == 2

    # player 1 income
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.INCOME)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 3
    assert env.players[1].coins == 3

    # player 0 assassinates player 1
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.ASSASSIN, target=1)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.players[0].coins == 0
    assert env.players[1].coins == 3
    assert env.status == AgentStatus.CAN_CHALLENGE

    # player 1 does not challenge
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.NO_CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.players[0].coins == 0
    assert env.players[1].coins == 3
    assert env.status == AgentStatus.ASSASSINATED

    # player 1 doesn't block
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.LET_ASSASSINATION)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.players[0].coins == 0
    assert env.players[1].coins == 3
    assert env.status == AgentStatus.LOSE_INFLUENCE

    # player 1 loses influence

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.LOSE_INFLUENCE, 0)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.players[0].coins == 0
    assert env.players[1].coins == 3
    assert env.status == AgentStatus.TURN
    assert env.players[1].hidden_cards == [Cards.CAPTAIN]
    assert env.players[1].visible_cards == [Cards.AMBASSADOR]


def test_assassination_challenge_succeeds():
    # 0 assassinates 1, 1 doesn't challenge; 1 loses one influence

    env = raw_env(render_mode="human", player_count=2)
    env.reset(seed=42)

    env.players[0].hidden_cards = [Cards.DUKE, Cards.AMBASSADOR]
    env.players[1].hidden_cards = [Cards.AMBASSADOR, Cards.CAPTAIN]

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # start with player 0 income
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.INCOME)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 3
    assert env.players[1].coins == 2

    # player 1 income
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.INCOME)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 3
    assert env.players[1].coins == 3

    # player 0 assassinates player 1
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.ASSASSIN, target=1)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.players[0].coins == 0
    assert env.players[1].coins == 3
    assert env.status == AgentStatus.CAN_CHALLENGE

    # player 1 challenges
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.players[0].coins == 0
    assert env.players[1].coins == 3
    assert env.status == AgentStatus.LOSE_INFLUENCE

    # player 1 loses influence (the duke)
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.LOSE_INFLUENCE, 0)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.players[0].coins == 0
    assert env.players[1].coins == 3
    assert env.status == AgentStatus.TURN
    assert env.players[0].hidden_cards == [Cards.AMBASSADOR]
    assert env.players[0].visible_cards == [Cards.DUKE]


def test_foreign_aid_succeeds():
    # 0 calls foreign aid, 1 lets it go through

    env = raw_env(render_mode="human", player_count=2)
    env.reset(seed=42)

    env.players[0].hidden_cards = [Cards.DUKE, Cards.AMBASSADOR]
    env.players[1].hidden_cards = [Cards.AMBASSADOR, Cards.CAPTAIN]

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # start with player 0 foreign aid
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.FOREIGN_AID)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.CAN_BLOCK_FOREIGN_AID
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 1 lets foreign aid go through

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.LET_FOREIGN_AID)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 1
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 4
    assert env.players[1].coins == 2


def test_foreign_aid_fails_duke_no_challenge():
    # 0 calls foreign aid, 1 uses duke, 0 doesn't challenge, nothing happens.

    env = raw_env(render_mode="human", player_count=2)
    env.reset(seed=42)

    env.players[0].hidden_cards = [Cards.DUKE, Cards.AMBASSADOR]
    env.players[1].hidden_cards = [Cards.AMBASSADOR, Cards.CAPTAIN]

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # start with player 0 foreign aid
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.FOREIGN_AID)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.CAN_BLOCK_FOREIGN_AID
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 1 blocks foreign aid

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.BLOCK_FOREIGN_AID)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.CAN_CHALLENGE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 0 doesn't challenge, the status quo remains

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.NO_CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 1
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2
    assert len(env.players[0].hidden_cards) == 2
    assert len(env.players[1].hidden_cards) == 2


def test_foreign_aid_fails_duke_challenge_fails():
    # 0 calls foreign aid, 1 uses duke, 0 challenges, 1 actually has duke so 0 loses influence

    env = raw_env(render_mode="human", player_count=2)
    env.reset(seed=42)

    env.players[0].hidden_cards = [Cards.DUKE, Cards.AMBASSADOR]
    env.players[1].hidden_cards = [Cards.AMBASSADOR, Cards.DUKE]

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # start with player 0 foreign aid
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.FOREIGN_AID)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.CAN_BLOCK_FOREIGN_AID
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 1 blocks foreign aid

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.BLOCK_FOREIGN_AID)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.CAN_CHALLENGE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 0 challenges incorrectly

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.LOSE_INFLUENCE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2
    assert len(env.players[0].hidden_cards) == 2
    assert len(env.players[1].hidden_cards) == 2

    # player 0 loses influence (the ambassador)
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.LOSE_INFLUENCE, 1)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 1
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2
    assert env.players[0].hidden_cards == [Cards.DUKE]
    assert env.players[0].visible_cards == [Cards.AMBASSADOR]
    assert len(env.players[1].hidden_cards) == 2


def test_foreign_aid_succeeds_duke_challenge_succeeds():
    # 0 calls foreign aid, 1 uses duke, 0 challenges, 1 doesn't have duke so 1 loses influence

    env = raw_env(render_mode="human", player_count=2)
    env.reset(seed=42)

    env.players[0].hidden_cards = [Cards.DUKE, Cards.AMBASSADOR]
    env.players[1].hidden_cards = [Cards.CAPTAIN, Cards.CONTESSA]

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # start with player 0 foreign aid
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.FOREIGN_AID)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.CAN_BLOCK_FOREIGN_AID
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 1 blocks foreign aid

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.BLOCK_FOREIGN_AID)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.CAN_CHALLENGE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 0 challenges correctly

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.LOSE_INFLUENCE
    assert env.players[0].coins == 4
    assert env.players[1].coins == 2
    assert len(env.players[0].hidden_cards) == 2
    assert len(env.players[1].hidden_cards) == 2

    # player 1 loses influence (the captain)
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.LOSE_INFLUENCE, 0)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 1
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 4
    assert env.players[1].coins == 2
    assert env.players[1].hidden_cards == [Cards.CONTESSA]
    assert env.players[1].visible_cards == [Cards.CAPTAIN]
    assert len(env.players[0].hidden_cards) == 2


def test_steal_succeeds_no_challenge():
    # 0 steals, 1 doesn't challenge or block, 1 loses 2 coins, 0 gains 2 coins

    env = raw_env(render_mode="human", player_count=2)
    env.reset(seed=42)

    env.players[0].hidden_cards = [Cards.AMBASSADOR, Cards.CAPTAIN]
    env.players[1].hidden_cards = [Cards.CONTESSA, Cards.DUKE]

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # start with player 0 steal
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CAPTAIN, target=1)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.CAN_CHALLENGE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 1 does not challenge

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.NO_CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.STOLEN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 1 does not block the steal
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.LET_STEAL)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 1
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 4
    assert env.players[1].coins == 0

    assert len(env.players[0].hidden_cards) == 2
    assert len(env.players[1].hidden_cards) == 2

    # player 1 just gets income
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.INCOME)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 4
    assert env.players[1].coins == 1

    assert len(env.players[0].hidden_cards) == 2
    assert len(env.players[1].hidden_cards) == 2


def test_steal_fails_challenge_success():
    # 0 steals, 1 challenges it, 0 doesn't have captain,

    env = raw_env(render_mode="human", player_count=2)
    env.reset(seed=42)

    env.players[0].hidden_cards = [Cards.AMBASSADOR, Cards.DUKE]
    env.players[1].hidden_cards = [Cards.CONTESSA, Cards.CAPTAIN]

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # start with player 0 steal
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CAPTAIN, target=1)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.CAN_CHALLENGE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 1 challenges correctly

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.LOSE_INFLUENCE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 0 loses influence (the ambassador)

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.LOSE_INFLUENCE, 0)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 1
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    assert env.players[0].hidden_cards == [Cards.DUKE]
    assert env.players[0].visible_cards == [Cards.AMBASSADOR]

    # player 1 just gets income
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.INCOME)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 3

    assert env.players[0].hidden_cards == [Cards.DUKE]
    assert env.players[0].visible_cards == [Cards.AMBASSADOR]
    assert len(env.players[1].hidden_cards) == 2


def test_steal_succeeds_challenge_fails():
    # 0 steals, 1 challenges it, 0 has captain, 1 loses influence

    env = raw_env(render_mode="human", player_count=2)
    env.reset(seed=42)

    env.players[0].hidden_cards = [Cards.AMBASSADOR, Cards.CAPTAIN]
    env.players[1].hidden_cards = [Cards.CONTESSA, Cards.DUKE]

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # start with player 0 steal
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CAPTAIN, target=1)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.CAN_CHALLENGE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 1 challenges incorrectly

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.LOSE_INFLUENCE
    assert env.players[0].coins == 4
    assert env.players[1].coins == 0

    # player 1 loses influence (the duke)

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.LOSE_INFLUENCE, 1)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 1
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 4
    assert env.players[1].coins == 0

    assert env.players[1].hidden_cards == [Cards.CONTESSA]
    assert env.players[1].visible_cards == [Cards.DUKE]
    assert len(env.players[0].hidden_cards) == 2

    # player 1 just gets income
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.INCOME)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 4
    assert env.players[1].coins == 1

    assert env.players[1].hidden_cards == [Cards.CONTESSA]
    assert env.players[1].visible_cards == [Cards.DUKE]
    assert len(env.players[0].hidden_cards) == 2


def test_steal_fails_blocked_by_ambassador():
    # 0 steals, 1 doesn't challenge, 1 blocks with ambassador, 0 doesn't challenge, nothing happens

    env = raw_env(render_mode="human", player_count=2)
    env.reset(seed=42)

    env.players[0].hidden_cards = [Cards.AMBASSADOR, Cards.CAPTAIN]
    env.players[1].hidden_cards = [Cards.CONTESSA, Cards.DUKE]

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # start with player 0 steal
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CAPTAIN, target=1)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.CAN_CHALLENGE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 1 doesn't challenge

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.NO_CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.STOLEN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 1 blocks the steal with ambassador

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.BLOCK_STEAL_WITH_AMBASSADOR)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.CAN_CHALLENGE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    assert len(env.players[0].hidden_cards) == 2
    assert len(env.players[1].hidden_cards) == 2

    # player 0 doesn't challenge, the status quo remains

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.NO_CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 1
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    assert len(env.players[0].hidden_cards) == 2
    assert len(env.players[1].hidden_cards) == 2

    # player 1 just gets income
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.INCOME)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 3

    assert len(env.players[0].hidden_cards) == 2
    assert len(env.players[1].hidden_cards) == 2


def test_steal_succeeds_ambassador_block_challenge_succeeds():
    # 0 steals, 1 doesn't challenge, 1 blocks with ambassador, 0 calls the bluff successfully, 1 loses influence

    env = raw_env(render_mode="human", player_count=2)
    env.reset(seed=42)

    env.players[0].hidden_cards = [Cards.AMBASSADOR, Cards.CAPTAIN]
    env.players[1].hidden_cards = [Cards.CONTESSA, Cards.DUKE]

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # start with player 0 steal
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CAPTAIN, target=1)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.CAN_CHALLENGE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 1 doesn't challenge

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.NO_CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.STOLEN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 1 blocks the steal with ambassador

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.BLOCK_STEAL_WITH_AMBASSADOR)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.CAN_CHALLENGE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    assert len(env.players[0].hidden_cards) == 2
    assert len(env.players[1].hidden_cards) == 2

    # player 0 challenges and succeeds

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.LOSE_INFLUENCE
    assert env.players[0].coins == 4
    assert env.players[1].coins == 0

    assert len(env.players[0].hidden_cards) == 2
    assert len(env.players[1].hidden_cards) == 2

    # player 1 loses influence (the duke)
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.LOSE_INFLUENCE, 1)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 1
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 4
    assert env.players[1].coins == 0

    assert len(env.players[0].hidden_cards) == 2

    assert env.players[1].hidden_cards == [Cards.CONTESSA]
    assert env.players[1].visible_cards == [Cards.DUKE]

    # player 1 just gets income
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.INCOME)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 4
    assert env.players[1].coins == 1

    assert len(env.players[0].hidden_cards) == 2
    assert env.players[1].hidden_cards == [Cards.CONTESSA]
    assert env.players[1].visible_cards == [Cards.DUKE]


def test_steal_succeeds_captain_block_challenge_succeeds():
    # 0 steals, 1 doesn't challenge, 1 blocks with captain, 0 calls the bluff successfully, 1 loses influence

    env = raw_env(render_mode="human", player_count=2)
    env.reset(seed=42)

    env.players[0].hidden_cards = [Cards.AMBASSADOR, Cards.CAPTAIN]
    env.players[1].hidden_cards = [Cards.CONTESSA, Cards.DUKE]

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # start with player 0 steal
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CAPTAIN, target=1)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.CAN_CHALLENGE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 1 doesn't challenge

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.NO_CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.STOLEN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 1 blocks the steal with ambassador

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.BLOCK_STEAL_WITH_CAPTAIN)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.CAN_CHALLENGE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    assert len(env.players[0].hidden_cards) == 2
    assert len(env.players[1].hidden_cards) == 2

    # player 0 challenges and succeeds

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.LOSE_INFLUENCE
    assert env.players[0].coins == 4
    assert env.players[1].coins == 0

    assert len(env.players[0].hidden_cards) == 2
    assert len(env.players[1].hidden_cards) == 2

    # player 1 loses influence (the duke)
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.LOSE_INFLUENCE, 1)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 1
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 4
    assert env.players[1].coins == 0

    assert len(env.players[0].hidden_cards) == 2

    assert env.players[1].hidden_cards == [Cards.CONTESSA]
    assert env.players[1].visible_cards == [Cards.DUKE]

    # player 1 just gets income
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.INCOME)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 4
    assert env.players[1].coins == 1

    assert len(env.players[0].hidden_cards) == 2
    assert env.players[1].hidden_cards == [Cards.CONTESSA]
    assert env.players[1].visible_cards == [Cards.DUKE]


def test_steal_fails_ambassador_block_challenge_fails():
    # 0 steals, 1 doesn't challenge, 1 blocks with ambassador, 0 challenges and fails, 0 loses influence

    env = raw_env(render_mode="human", player_count=2)
    env.reset(seed=42)

    env.players[0].hidden_cards = [Cards.AMBASSADOR, Cards.CAPTAIN]
    env.players[1].hidden_cards = [Cards.AMBASSADOR, Cards.DUKE]

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # start with player 0 steal
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CAPTAIN, target=1)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.CAN_CHALLENGE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 1 doesn't challenge

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.NO_CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.STOLEN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 1 blocks the steal with ambassador

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.BLOCK_STEAL_WITH_AMBASSADOR)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.CAN_CHALLENGE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    assert len(env.players[0].hidden_cards) == 2
    assert len(env.players[1].hidden_cards) == 2

    # player 0 challenges and fails

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.LOSE_INFLUENCE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    assert len(env.players[0].hidden_cards) == 2
    assert len(env.players[1].hidden_cards) == 2

    # player 0 loses influence (the captain)
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.LOSE_INFLUENCE, 1)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 1
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    assert env.players[0].hidden_cards == [Cards.AMBASSADOR]
    assert env.players[0].visible_cards == [Cards.CAPTAIN]

    # player 1 just gets income
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.INCOME)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 3

    assert len(env.players[1].hidden_cards) == 2

    assert env.players[0].hidden_cards == [Cards.AMBASSADOR]
    assert env.players[0].visible_cards == [Cards.CAPTAIN]


def test_steal_fails_captain_block_challenge_fails():
    # 0 steals, 1 doesn't challenge, 1 blocks with captain, 0 challenges and fails, 0 loses influence

    env = raw_env(render_mode="human", player_count=2)
    env.reset(seed=42)

    env.players[0].hidden_cards = [Cards.AMBASSADOR, Cards.CAPTAIN]
    env.players[1].hidden_cards = [Cards.CAPTAIN, Cards.DUKE]

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # start with player 0 steal
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CAPTAIN, target=1)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.CAN_CHALLENGE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 1 doesn't challenge

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.NO_CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.STOLEN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 1 blocks the steal with ambassador

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.BLOCK_STEAL_WITH_CAPTAIN)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.CAN_CHALLENGE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    assert len(env.players[0].hidden_cards) == 2
    assert len(env.players[1].hidden_cards) == 2

    # player 0 challenges and fails

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.LOSE_INFLUENCE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    assert len(env.players[0].hidden_cards) == 2
    assert len(env.players[1].hidden_cards) == 2

    # player 0 loses influence (the captain)
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.LOSE_INFLUENCE, 1)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 1
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    assert env.players[0].hidden_cards == [Cards.AMBASSADOR]
    assert env.players[0].visible_cards == [Cards.CAPTAIN]

    # player 1 just gets income
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.INCOME)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 3

    assert len(env.players[1].hidden_cards) == 2

    assert env.players[0].hidden_cards == [Cards.AMBASSADOR]
    assert env.players[0].visible_cards == [Cards.CAPTAIN]


def test_ambassador_succeeds_no_challenge():
    # 0 ambassadors, 1 doesn't challenge, 0 gets new cards

    env = raw_env(render_mode="human", player_count=2)
    env.reset(seed=42)

    env.players[0].hidden_cards = [Cards.AMBASSADOR, Cards.CAPTAIN]
    env.players[1].hidden_cards = [Cards.CAPTAIN, Cards.DUKE]

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # start with player 0 ambassador
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.AMBASSADOR)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.CAN_CHALLENGE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 1 doesn't challenge

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.NO_CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.AMBASSADOR_EXCHANGE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 0 exchanges

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.AMBASSADOR_EXCHANGE_4, 1)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 1
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    assert len(env.players[0].hidden_cards) == 2
    assert len(env.players[1].hidden_cards) == 2


def test_ambassador_comprehensive():
    # 0 ambassadors, 1 doesn't challenge, 0 gets new cards

    env = raw_env(render_mode="human", player_count=2)
    env.reset(seed=42)

    env.players[0].hidden_cards = [Cards.DUKE, Cards.CAPTAIN]
    env.players[1].hidden_cards = [Cards.AMBASSADOR, Cards.CONTESSA]

    assert env.agent_selection == "0"
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # start with player 0 ambassador
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.AMBASSADOR)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.CAN_CHALLENGE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 1 challenges and succeeds

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.LOSE_INFLUENCE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    # player 0 loses influence (the duke)

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.LOSE_INFLUENCE, 0)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 1
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 2

    assert len(env.players[0].hidden_cards) == 1
    assert env.players[0].hidden_cards == [Cards.CAPTAIN]
    assert env.players[0].visible_cards == [Cards.DUKE]
    assert len(env.players[1].hidden_cards) == 2

    # player 1 takes income

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.INCOME)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.TURN
    assert env.players[0].coins == 2
    assert env.players[1].coins == 3

    assert len(env.players[0].hidden_cards) == 1
    assert env.players[0].hidden_cards == [Cards.CAPTAIN]
    assert env.players[0].visible_cards == [Cards.DUKE]
    assert len(env.players[1].hidden_cards) == 2

    # player 0 ambassadors again (but actually has ambassador because I hacked it)
    env.players[0].hidden_cards = [Cards.AMBASSADOR]

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.AMBASSADOR)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.CAN_CHALLENGE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 3

    # player 1 challenges and fails

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.CHALLENGE)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.LOSE_INFLUENCE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 3

    # player 1 loses influence (the duke)

    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.LOSE_INFLUENCE, 0)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "0"
    assert env.turn_agent == 0
    assert env.status == AgentStatus.AMBASSADOR_EXCHANGE
    assert env.players[0].coins == 2
    assert env.players[1].coins == 3

    assert len(env.players[0].hidden_cards) == 1
    assert env.players[0].visible_cards == [Cards.DUKE]
    assert env.players[1].hidden_cards == [Cards.CONTESSA]
    assert env.players[1].visible_cards == [Cards.AMBASSADOR]

    # player 0 exchanges
    agent = env.players[int(env.agent_selection)]
    observation, reward, termination, truncation, info = env.last()
    env.current_actions(agent)
    action_spec = ActionSpecification(Action.AMBASSADOR_EXCHANGE_3, 1)
    action = env.action_specification_to_index(action_spec)
    env.step(action)  # type: ignore

    assert env.agent_selection == "1"
    assert env.turn_agent == 1

    assert env.players[0].coins == 2
    assert env.players[1].coins == 3

    assert env.players[0].visible_cards == [Cards.DUKE]
    assert env.players[1].hidden_cards == [Cards.CONTESSA]
    assert env.players[1].visible_cards == [Cards.AMBASSADOR]


test_steal_succeeds_challenge_fails()
