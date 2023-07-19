import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

import random
from agent import Agent
import agent
from typing import List, Tuple, Dict, Optional
from utils import (
    SPECIFY_OPPONENT,
    ActionSpecification,
    AgentStatus,
    Cards,
    Action,
    IS_FULLY_SPECIFIED,
    ACTION_TO_CARD,
    map_combinations,
    CARD_TO_STRING,
)


# CARD_NAMES = ["AMBASSADOR", "ASSASSIN", "CAPTAIN", "CONTESSA", "DUKE"]


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, player_count: int = 6, render_mode=None, verbose: bool = False):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed açfter initialization.
        """
        assert (
            player_count <= 6 and player_count >= 2
        ), "player_count must be between 2 and 6"

        self.possible_agents = [str(i) for i in range(player_count)]
        self.render_mode = render_mode
        self.player_count = player_count
        self.verbose = verbose

        self.ambassador_mapping_3 = map_combinations(3, 1)
        self.ambassador_mapping_4 = map_combinations(4, 2)

    def pull_card(self) -> Cards:
        index = random.randint(
            0, len(self.deck) - 1
        )  # instead, randomly choose a card from the center using Card
        card = self.deck[index]
        del self.deck[index]
        return card

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    # @functools.lru_cache(maxsize=None)

    def observe(self, agent_id: str):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        player = self.players[int(agent_id)]
        observation = []
        for i in range(self.player_count):
            if self.players[i] == player:
                observation.extend(player.hidden_cards) 
                observation.extend(player.visible_cards)
            else:
                # extend observation with -1 * len(self.players[i].hidden_cards)
                observation.extend([-1] * len(self.players[i].hidden_cards))
                observation.extend(self.players[i].visible_cards)

        observation.extend([self.players[i].coins for i in range(self.player_count)])

        assert len(observation) == 3 * self.player_count, "observation is wrong length"
        player.observation = observation
        return np.array(observation)

    def observation_space(self, agent: Agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/

        # first 2 * player_count values are the cards of all players
        # next player_count values are the coins of all players

        observation = []
        for i in range(self.player_count):
            if self.players[i] == agent:
                observation.extend(agent.hidden_cards)
                observation.extend(agent.visible_cards)
            else:
                observation.extend(self.players[i].hidden_cards)

        observation.extend([self.players[i].coins for i in range(self.player_count)])

        return Discrete(len(observation))

    def current_actions_to_string(self, agent: Agent):
        string = ""
        for i, action in enumerate(agent.current_actions):
            string += f"{i}: "
            if action.name == Action.AMBASSADOR:
                string += "claim ambassador"
            elif action.name == Action.ASSASSIN:
                string += (
                    f"claim assassin, attempting to assassinate player {action.target}"
                )
            elif action.name == Action.CAPTAIN:
                string += (
                    f"claim captain, attempting to steal from player {action.target}"
                )
            elif action.name == Action.DUKE:
                string += "claim duke"
            elif action.name == Action.INCOME:
                string += "take income"
            elif action.name == Action.FOREIGN_AID:
                string += "attempt to take foreign aid"
            elif action.name == Action.COUP:
                string += f"coup player {action.target}"
            elif action.name == Action.CHALLENGE:
                string += f"challenge"
            elif action.name == Action.NO_CHALLENGE:
                string += f"don't challenge"
            elif action.name == Action.BLOCK_FOREIGN_AID:
                string += f"block foreign aid"
            elif action.name == Action.LET_FOREIGN_AID:
                string += f"let foreign aid occur"
            elif action.name == Action.BLOCK_STEAL_WITH_AMBASSADOR:
                string += f"block steal with ambassador"
            elif action.name == Action.BLOCK_STEAL_WITH_CAPTAIN:
                string += f"block steal with captain"
            elif action.name == Action.LET_STEAL:
                string += f"let steal occur"
            elif action.name == Action.BLOCK_ASSASSINATION:
                string += f"block assassination by claiming contessa"
            elif action.name == Action.LET_ASSASSINATION:
                string += f"let assassination occur onto yourself"
            elif action.name == Action.LOSE_INFLUENCE:
                string += f"lose a card: {CARD_TO_STRING[agent.hidden_cards[action.target]]}"  # type: ignore
            elif action.name == Action.AMBASSADOR_EXCHANGE_3:
                cards = agent.hidden_cards + agent.ambassador_center_view  # type: ignore
                keep_cards_indices = self.ambassador_mapping_3[action.target]  # type: ignore
                keep_cards = [cards[index] for index in keep_cards_indices]  # type: ignore
                keep_cards_string = ", ".join(
                    [CARD_TO_STRING[card] for card in keep_cards]
                )

                string += f"keep cards " + keep_cards_string
            elif action.name == Action.AMBASSADOR_EXCHANGE_4:
                cards = agent.hidden_cards + agent.ambassador_center_view  # type: ignore
                keep_cards_indices = self.ambassador_mapping_4[action.target]  # type: ignore
                keep_cards = [cards[index] for index in keep_cards_indices]  # type: ignore
                keep_cards_string = ", ".join(
                    [CARD_TO_STRING[card] for card in keep_cards]
                )

                string += f"ambassador - keep " + keep_cards_string
            string += "\n"
        return string

    def current_actions(self, agent: Agent):
        assert self.status is not None, "status must be specified"

        agent.current_actions = []

        if self.status == AgentStatus.TURN:
            actions = [
                Action.INCOME,
                Action.FOREIGN_AID,
                Action.AMBASSADOR,
                Action.CAPTAIN,
                Action.DUKE,
                Action.ASSASSIN
            ]
            if agent.coins >= 3:
                actions.append(Action.ASSASSIN)
            if agent.coins >= 7:
                actions.append(Action.COUP)

            if agent.coins >= 10:
                actions = [Action.COUP]
        elif self.status == AgentStatus.STOLEN:
            actions = [
                Action.BLOCK_STEAL_WITH_AMBASSADOR,
                Action.BLOCK_STEAL_WITH_CAPTAIN,
                Action.LET_STEAL,
            ]
        elif self.status == AgentStatus.AMBASSADOR_EXCHANGE:
            if len(agent.hidden_cards) == 2:
                actions = [Action.AMBASSADOR_EXCHANGE_4]
            elif len(agent.hidden_cards) == 1:
                actions = [Action.AMBASSADOR_EXCHANGE_3]
            else:
                raise Exception("Invalid number of cards")
        elif self.status == AgentStatus.ASSASSINATED:
            actions = [Action.BLOCK_ASSASSINATION, Action.LET_ASSASSINATION]
        elif self.status == AgentStatus.LOSE_INFLUENCE:
            actions = [Action.LOSE_INFLUENCE]
        elif self.status == AgentStatus.CAN_CHALLENGE:
            actions = [Action.CHALLENGE, Action.NO_CHALLENGE]
        elif self.status == AgentStatus.CAN_BLOCK_FOREIGN_AID:
            actions = [Action.BLOCK_FOREIGN_AID, Action.LET_FOREIGN_AID]
        else:
            raise Exception("Invalid status")

        for action in actions:
            if action == Action.AMBASSADOR_EXCHANGE_3:
                for i in range(len(self.ambassador_mapping_3)):
                    agent.current_actions.append(ActionSpecification(action, i))
            elif action == Action.AMBASSADOR_EXCHANGE_4:
                for i in range(len(self.ambassador_mapping_4)):
                    agent.current_actions.append(ActionSpecification(action, i))
            elif IS_FULLY_SPECIFIED[action]:
                agent.current_actions.append(ActionSpecification(action, None))
            else:
                if SPECIFY_OPPONENT[action]:
                    for target in range(self.player_count):
                        if (
                            str(target) != agent.name
                            and not self.terminations[str(target)]
                        ):
                            agent.current_actions.append(
                                ActionSpecification(action, target)
                            )
                else:
                    for card in range(len(agent.hidden_cards)):
                        agent.current_actions.append(ActionSpecification(action, card))

    def action_specification_to_index(
        self, action_spec: ActionSpecification, player: Optional[Agent] = None
    ):
        if player is None:
            player = self.players[int(self.agent_selection)]
        for i in range(len(player.current_actions)):
            if (
                player.current_actions[i].name == action_spec.name
                and player.current_actions[i].target == action_spec.target
            ):
                return i

        raise Exception("action_spec not found in current_actions")

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    # @functools.lru_cache(maxsize=None)
    def action_space(self, agent: Agent) -> Discrete:
        return Discrete(len(agent.current_actions))

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        # if self.render_mode is None:
        #     gymnasium.logger.warn(
        #         "You are calling render method without specifying any render mode."
        #     )
        #     return

        # if len(self.agents) == 2:
        #     string = "Current state: Agent1: {} , Agent2: {}".format(
        #         MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]
        #     )
        # else:
        #     string = "Game over"

        pass

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}  # type: ignore

        # self.agent_selection = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self.agent_selection = str(0)
        self.turn_agent: int = 0  # if we resolved this turn, who is going right now? +1 to get the agent which goes next

        self.status: AgentStatus = AgentStatus.TURN
        self.action_to_challenge: Optional[ActionSpecification] = None

        # self.deck should contain 3 of each card
        # loop through all values in Cards enum and add 3 of each card to the deck
        self.deck = []
        for card in Cards:
            for i in range(3):
                self.deck.append(card)

        self.players: List[Agent] = []
        self.blocking_agent: Optional[int] = None

        self.alive_count = self.player_count
        self.challenging_players: List[str] = []
        self.blocking_players: List[str] = []

        # adding players to the game
        for i in range(0, self.player_count):
            # pull 2 cards from the deck and deal to each player
            cards = []
            cards.append(self.pull_card())
            cards.append(self.pull_card())

            # create a Agent
            player = Agent(cards, 2, str(i), True)
            self.players.append(player)

        self.center: List[Cards] = []
        for i in range(3):
            self.center.append(self.pull_card())
        self.done = False
        self.history = ""

    def add_to_history(self, string: str, ambassador=False):
        if self.verbose:
            print(string)
        if ambassador:
            self.history += (
                f"Player {self.agent_selection} uses ambassador, seeing two centering cards and potentially exchanging"
                + "\n"
            )
            for i in range(self.player_count):
                if i == int(self.agent_selection):
                    self.players[i].history += string + "\n"
                else:
                    self.players[i].history += (
                        f"Player {self.agent_selection} uses ambassador, seeing two centering cards and potentially exchanging"
                        + "\n"
                    )
        else:
            self.history += string + "\n"
            for i in range(self.player_count):
                self.players[i].history += string + "\n"

    def step(self, action: int):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """


        player = self.players[int(self.agent_selection)]
        # action is a number, so we need to retrieve the action specification

        action_spec = player.current_actions[action]

        # if action_spec.name in ACTION_TO_CARD and ACTION_TO_CARD[action_spec.name] not in player.hidden_cards:


        if self.verbose:
            print(
                self.turn_agent,
                self.agent_selection,
                action,
                action_spec.name,
                action_spec.target,
                self.observe(self.agent_selection),
            )
        if action_spec.name == Action.AMBASSADOR_EXCHANGE_3:
            assert len(player.hidden_cards) == 1, "player must have 1 hidden card"
        elif action_spec.name == Action.AMBASSADOR_EXCHANGE_4:
            assert len(player.hidden_cards) == 2, "player must have 2 hidden cards"
        elif IS_FULLY_SPECIFIED[action_spec.name]:
            assert action_spec.target is None, "action_spec.target must be None"
        else:
            assert action_spec.target is not None, "action_spec.target must not be None"

        if action_spec.name == Action.INCOME:
            self.add_to_history(f"Player {self.agent_selection} takes income")
            player.coins += 1

            self.status = AgentStatus.TURN
            observation = self.observe(self.agent_selection)

            self.agent_selection = str(self.next_alive_agent(int(self.agent_selection)))
            self.turn_agent = self.next_alive_agent(self.turn_agent)

        elif action_spec.name == Action.FOREIGN_AID:
            self.add_to_history(
                f"Player {self.agent_selection} attempts to take foreign aid"
            )
            self.status = AgentStatus.CAN_BLOCK_FOREIGN_AID
            self.agent_selection = str(1) if self.turn_agent == 0 else str(0)

        elif action_spec.name in [Action.BLOCK_FOREIGN_AID, Action.LET_FOREIGN_AID]:
            if action_spec.name == Action.BLOCK_FOREIGN_AID:
                self.blocking_players.append(self.agent_selection)

            last_alive_index = max(
                [i for i, x in enumerate(self.terminations.values()) if not x]
            )

            if self.agent_selection == str(last_alive_index) or (
                self.agent_selection == str(last_alive_index - 1)
                and last_alive_index == self.turn_agent
            ):
                if len(self.blocking_players) == 0:
                    self.add_to_history("No one blocks foreign aid")
                    self.players[self.turn_agent].coins += 2

                    self.turn_agent = self.next_alive_agent(self.turn_agent)
                    self.agent_selection = str(self.turn_agent)
                    self.status = AgentStatus.TURN

                else:
                    self.blocking_agent = int(random.choice(self.blocking_players))
                    self.add_to_history(
                        f"Player {self.blocking_agent} blocks foreign aid"
                    )

                    self.blocking_players = []
                    self.status = AgentStatus.CAN_CHALLENGE
                    self.action_to_challenge = ActionSpecification(
                        Action.BLOCK_FOREIGN_AID
                    )
                    self.agent_selection = str(self.turn_agent)

        elif action_spec.name == Action.COUP:
            self.add_to_history(
                f"Player {self.agent_selection} coups player {action_spec.target}"
            )

            player.coins -= 7
            self.status = AgentStatus.LOSE_INFLUENCE
            self.agent_selection = str(action_spec.target)

        elif action_spec.name == Action.LOSE_INFLUENCE:
            self.add_to_history(f"Player {self.agent_selection} loses influence: card {action_spec.target} ({CARD_TO_STRING[player.hidden_cards[action_spec.target]]})")  # type: ignore

            card = player.hidden_cards[action_spec.target]  # type: ignore
            del player.hidden_cards[action_spec.target]  # type: ignore
            player.visible_cards.append(card)

            if len(player.hidden_cards) == 0:
                self.terminations[self.agent_selection] = True
                self.alive_count -= 1
                if self.alive_count == 1:
                    self.done = True
                    return
            if self.action_to_challenge and self.action_to_challenge.name == Action.AMBASSADOR and self.turn_agent != int(self.agent_selection):  # type: ignore
                self.status = AgentStatus.AMBASSADOR_EXCHANGE
                self.agent_selection = str(self.turn_agent)
            else:
                self.status = AgentStatus.TURN

                self.turn_agent = self.next_alive_agent(self.turn_agent)
                self.agent_selection = str(self.turn_agent)

        elif action_spec.name == Action.LET_ASSASSINATION:
            self.add_to_history(
                f"Player {self.agent_selection} lets assassination occur"
            )
            self.status = AgentStatus.LOSE_INFLUENCE

        elif action_spec.name == Action.LET_STEAL:
            self.add_to_history(f"Player {self.agent_selection} lets steal occur")
            self.players[self.turn_agent].coins += 2
            self.players[int(self.agent_selection)].coins -= 2

            self.status = AgentStatus.TURN
            self.turn_agent = self.next_alive_agent(self.turn_agent)
            self.agent_selection = str(self.turn_agent)

        elif action_spec.name == Action.AMBASSADOR_EXCHANGE_3:
            assert len(player.hidden_cards) == 1, "player must have 1 hidden card"

            cards = player.hidden_cards + self.players[self.turn_agent].ambassador_center_view  # type: ignore
            keep = self.ambassador_mapping_3[action_spec.target]  # type: ignore

            self.add_to_history(
                f"Player {self.agent_selection} sees cards: {[CARD_TO_STRING[card] for card in cards]}; keeps cards {[CARD_TO_STRING[cards[index]] for index in keep]}",
                ambassador=True,
            )

            player.hidden_cards = [cards[keep[0]]]
            cards = cards[1:]

            assert len(cards) == 2, "cards must have 2 elements"
            self.center[-2:] = cards

            # shuffle the center
            random.shuffle(self.center)

            # set the ambassador center view to None
            self.players[self.turn_agent].ambassador_center_view = None

            self.turn_agent = self.next_alive_agent(self.turn_agent)
            self.agent_selection = str(self.turn_agent)
            self.status = AgentStatus.TURN

        elif action_spec.name == Action.AMBASSADOR_EXCHANGE_4:
            assert len(player.hidden_cards) == 2, "player must have 1 hidden card"

            cards = player.hidden_cards + self.players[self.turn_agent].ambassador_center_view  # type: ignore
            keep = self.ambassador_mapping_4[action_spec.target]  # type: ignore

            self.add_to_history(
                f"Player {self.agent_selection} sees cards: {[CARD_TO_STRING[card] for card in cards]}; keeps cards {[CARD_TO_STRING[cards[index]] for index in keep]}",
                ambassador=True,
            )

            player.hidden_cards = [cards[keep[0]], cards[keep[1]]]
            cards = cards[2:]

            assert len(cards) == 2, "cards must have 2 elements"
            self.center[-2:] = cards

            # shuffle the center
            random.shuffle(self.center)

            # set the ambassador center view to None
            self.players[self.turn_agent].ambassador_center_view = None

            self.turn_agent = self.next_alive_agent(self.turn_agent)
            self.agent_selection = str(self.turn_agent)
            self.status = AgentStatus.TURN

        elif (
            action_spec.name == Action.DUKE
            or action_spec.name == Action.ASSASSIN
            or action_spec.name == Action.BLOCK_ASSASSINATION
            or action_spec.name == Action.CAPTAIN
            or action_spec.name == Action.BLOCK_STEAL_WITH_AMBASSADOR
            or action_spec.name == Action.BLOCK_STEAL_WITH_CAPTAIN
            or action_spec.name == Action.AMBASSADOR
        ):
            if action_spec.name == Action.ASSASSIN:
                self.players[self.turn_agent].coins -= 3

            # printing stuff
            if action_spec.name == Action.DUKE:
                self.add_to_history(f"Player {self.agent_selection} claims Duke")
            elif action_spec.name == Action.ASSASSIN:
                self.add_to_history(f"Player {self.agent_selection} claims Assassin")
            elif action_spec.name == Action.BLOCK_ASSASSINATION:
                self.add_to_history(f"Player {self.agent_selection} claims Contessa")
            elif action_spec.name == Action.CAPTAIN:
                self.add_to_history(
                    f"Player {self.agent_selection} claims Captain to steal from player {action_spec.target}"
                )

            elif action_spec.name == Action.BLOCK_STEAL_WITH_AMBASSADOR:
                self.add_to_history(
                    f"Player {self.agent_selection} claims Ambassador to block steal"
                )

            elif action_spec.name == Action.BLOCK_STEAL_WITH_CAPTAIN:
                self.add_to_history(
                    f"Player {self.agent_selection} claims Ambassador to block steal"
                )
            elif action_spec.name == Action.AMBASSADOR:
                self.add_to_history(f"Player {self.agent_selection} claims Ambassador")

            # end printing stuff

            self.status = AgentStatus.CAN_CHALLENGE
            self.action_to_challenge = action_spec
            if (
                action_spec.name == Action.BLOCK_ASSASSINATION
                or action_spec.name == Action.BLOCK_STEAL_WITH_AMBASSADOR
                or action_spec.name == Action.BLOCK_STEAL_WITH_CAPTAIN
            ):
                self.blocking_agent = int(self.agent_selection)
                self.agent_selection = str(self.turn_agent)
            else:
                self.agent_selection = str(1) if self.turn_agent == 0 else str(0)

        elif action_spec.name in [Action.CHALLENGE, Action.NO_CHALLENGE]:
            if self.action_to_challenge.name == Action.BLOCK_FOREIGN_AID:  # type: ignore
                if action_spec.name == Action.CHALLENGE:
                    lose_influence = str(self.turn_agent) if Cards.DUKE in self.players[int(self.blocking_agent)].hidden_cards else self.blocking_agent  # type: ignore

                    if lose_influence == str(self.turn_agent):
                        self.add_to_history(
                            f"Player {self.turn_agent} incorrectly challenges player {self.turn_agent} on the duke, so the foreign aid fails and player {self.turn_agent} loses a card."
                        )
                    else:
                        self.add_to_history(
                            f"Player {self.turn_agent} correctly challenges player {self.blocking_agent} on the duke, so the foreign aid succeeds and player {self.blocking_agent} loses a card."
                        )

                    assert (
                        lose_influence is not None
                    ), "lose_influence must not be None in FOREIGN_AID"

                    self.status = AgentStatus.LOSE_INFLUENCE
                    if lose_influence == self.blocking_agent:
                        self.agent_selection = str(self.blocking_agent)
                        self.players[self.turn_agent].coins += 2
                    else:
                        self.agent_selection = str(self.turn_agent)

                else:
                    self.add_to_history(
                        f"Player {self.turn_agent} does not challenge, so the foreign aid is blocked."
                    )
                    self.status = AgentStatus.TURN
                    self.turn_agent = self.next_alive_agent(self.turn_agent)
                    self.agent_selection = str(self.turn_agent)

                self.blocking_agent = None

            else:
                if action_spec.name == Action.CHALLENGE:
                    self.challenging_players.append(self.agent_selection)

                # get the maximum index whose element is False in self.terminations:
                last_alive_index = max(
                    [i for i, x in enumerate(self.terminations.values()) if not x]
                )
                if (
                    self.agent_selection == str(last_alive_index)
                    or (
                        self.agent_selection == str(last_alive_index - 1)
                        and last_alive_index == self.turn_agent
                    )
                    or self.action_to_challenge.name == Action.BLOCK_ASSASSINATION  # type: ignore
                    or self.action_to_challenge.name == Action.BLOCK_STEAL_WITH_AMBASSADOR  # type: ignore
                    or self.action_to_challenge.name == Action.BLOCK_STEAL_WITH_CAPTAIN  # type: ignore
                    or self.action_to_challenge.name == Action.AMBASSADOR  # type: ignore
                ):
                    lose_influence = self.resolve_challenge(
                        self.challenging_players, self.action_to_challenge  # type: ignore
                    )

                    if lose_influence is None:
                        self.add_to_history(f"No one challenges")

                        if self.action_to_challenge.name == Action.DUKE:  # type: ignore
                            self.add_to_history(
                                f"Player {self.turn_agent} successfully gains 3 coins"
                            )
                            self.players[self.turn_agent].coins += 3
                            self.turn_agent = self.next_alive_agent(self.turn_agent)
                            self.agent_selection = str(self.turn_agent)
                            self.status = AgentStatus.TURN
                        elif self.action_to_challenge.name == Action.ASSASSIN:  # type: ignore
                            # check if contessa
                            self.status = AgentStatus.ASSASSINATED
                            self.agent_selection = str(self.action_to_challenge.target)  # type: ignore

                        elif self.action_to_challenge.name == Action.CAPTAIN:  # type: ignore
                            self.status = AgentStatus.STOLEN
                            self.agent_selection = str(self.action_to_challenge.target)  # type: ignore

                        elif self.action_to_challenge.name == Action.BLOCK_STEAL_WITH_AMBASSADOR or self.action_to_challenge.name == Action.BLOCK_STEAL_WITH_CAPTAIN or self.action_to_challenge.name == Action.BLOCK_ASSASSINATION:  # type: ignore
                            self.turn_agent = self.next_alive_agent(self.turn_agent)
                            self.agent_selection = str(self.turn_agent)
                            self.status = AgentStatus.TURN

                        elif self.action_to_challenge.name == Action.AMBASSADOR:  # type: ignore
                            self.status = AgentStatus.AMBASSADOR_EXCHANGE
                            self.agent_selection = str(self.turn_agent)

                            random.shuffle(self.center)
                            # take the last 2 cards from the center
                            self.players[
                                self.turn_agent
                            ].ambassador_center_view = self.center[-2:]

                    elif (self.blocking_agent is None and lose_influence == str(self.turn_agent)) or (self.blocking_agent is not None and lose_influence == str(self.blocking_agent)):  # type: ignore
                        # this means the challenged player loses influence
                        if (
                            self.agent_selection == str(self.turn_agent)
                            and self.action_to_challenge.name  # type: ignore
                            == Action.BLOCK_ASSASSINATION
                        ):
                            self.add_to_history(
                                f"The contessa was successfully challenged; immediately die."
                            )

                            player.visible_cards.extend(player.hidden_cards)
                            player.hidden_cards = []
                            self.terminations[str(self.agent_selection)] = True
                            self.alive_count -= 1

                            if self.alive_count == 1:
                                self.done = True
                                return

                            self.turn_agent = self.next_alive_agent(self.turn_agent)
                            self.agent_selection = str(self.turn_agent)
                            self.status = AgentStatus.TURN
                        else:
                            challenged_agent = (
                                self.blocking_agent
                                if self.blocking_agent is not None
                                else self.turn_agent
                            )
                            self.add_to_history(
                                f"Player {challenged_agent} is successfully challenged, so they lose a card."
                            )
                            self.status = AgentStatus.LOSE_INFLUENCE
                            self.agent_selection = str(challenged_agent)

                            if (
                                self.action_to_challenge.name  # type: ignore
                                == Action.BLOCK_STEAL_WITH_AMBASSADOR
                                or self.action_to_challenge.name  # type: ignore
                                == Action.BLOCK_STEAL_WITH_CAPTAIN
                            ):
                                self.players[self.turn_agent].coins += 2
                                self.players[self.blocking_agent].coins -= 2  # type: ignore

                                self.status = AgentStatus.LOSE_INFLUENCE
                                assert self.agent_selection == str(
                                    self.blocking_agent
                                ), "agent_selection must be blocking_agent"

                    else:
                        # the challenging player loses influence
                        if (
                            self.agent_selection != str(self.turn_agent)
                            and self.action_to_challenge.name == Action.ASSASSIN  # type: ignore
                            and Cards.CONTESSA
                            not in player.hidden_cards  # TODO: this should be the targeted player, not player
                        ):  # type: ignore  # if challenging assassin wrong, die if no contessa.
                            self.add_to_history(
                                f"Player actually has an assassin, so the challenge fails and they immediately die"
                            )
                            player.visible_cards.extend(player.hidden_cards)
                            player.hidden_cards = []
                            self.terminations[str(self.agent_selection)] = True
                            self.alive_count -= 1

                            if self.alive_count == 1:
                                self.done = True
                                return

                            self.turn_agent = self.next_alive_agent(self.turn_agent)
                            self.agent_selection = str(self.turn_agent)
                            self.status = AgentStatus.TURN

                        else:
                            challenged_agent = (
                                self.blocking_agent
                                if self.blocking_agent is not None
                                else self.turn_agent
                            )
                            self.add_to_history(
                                f"Player {lose_influence} tried to challenge, but was unsuccessful. They lose a card, and player {challenged_agent} gets to shuffle in a new card from the deck."
                            )

                            # the challenger loses influence
                            if self.action_to_challenge.name == Action.DUKE:  # type: ignore
                                self.add_to_history(
                                    f"Player {challenged_agent} successfully gains 3 coins"
                                )
                                self.players[challenged_agent].coins += 3

                            elif self.action_to_challenge.name == Action.CAPTAIN:  # type: ignore
                                self.add_to_history(
                                    f"Player {self.turn_agent} successfully steals 2 coins"
                                )
                                self.players[self.turn_agent].coins += 2
                                if self.blocking_agent:
                                    self.players[self.blocking_agent].coins -= 2  # type: ignore
                                else:
                                    self.players[int(self.agent_selection)].coins -= 2

                            elif (
                                self.action_to_challenge.name  # type: ignore
                                == Action.BLOCK_STEAL_WITH_AMBASSADOR
                                or self.action_to_challenge.name  # type: ignore
                                == Action.BLOCK_STEAL_WITH_CAPTAIN
                            ):
                                self.add_to_history(
                                    f"Player {self.turn_agent} incorrectly challenges the blocked steal, so they lose a card"
                                )
                                self.status = AgentStatus.LOSE_INFLUENCE
                                self.agent_selection = str(self.turn_agent)

                            elif self.action_to_challenge.name == Action.AMBASSADOR:  # type: ignore
                                self.status = AgentStatus.AMBASSADOR_EXCHANGE
                                self.agent_selection = str(self.turn_agent)

                                random.shuffle(self.center)
                                # take the last 2 cards from the center
                                self.players[
                                    self.turn_agent
                                ].ambassador_center_view = self.center[-2:]

                            # the challenged player gets a random card
                            # if self.blocking_agent:
                            #     challenged_agent = self.players[self.blocking_agent] # type: ignore
                            # else:
                            #     challenged_agent = self.players[self.turn_agent]

                            card_index = self.players[challenged_agent].hidden_cards.index(ACTION_TO_CARD[self.action_to_challenge.name])  # type: ignore

                            shuffled_cards = [
                                self.players[challenged_agent].hidden_cards[card_index]
                            ] + self.center
                            random.shuffle(shuffled_cards)
                            self.players[challenged_agent].hidden_cards[
                                card_index
                            ] = shuffled_cards[0]
                            self.center = shuffled_cards[1:]

                            self.status = AgentStatus.LOSE_INFLUENCE
                            self.agent_selection = str(lose_influence)

                    self.challenging_players = []

                    self.blocking_agent = None
                else:
                    self.agent_selection = str(
                        self.next_alive_agent(int(self.agent_selection))
                    )
                    if self.agent_selection == str(self.turn_agent):
                        # skip the turn agent
                        self.agent_selection = str(
                            self.next_alive_agent(int(self.agent_selection))
                        )

            self.action_to_challenge = None

    def next_alive_agent(self, i: int) -> int:
        """
        returns next alive agent
        """
        i_old = i
        i = (i + 1) % self.player_count
        while self.terminations[str(i)]:
            i = (i + 1) % self.player_count
        assert i != i_old, "there is only one remaining player"
        return i

    def resolve_challenge(
        self, challenging_players: List[str], action_to_challenge: ActionSpecification
    ) -> Optional[str]:
        """
        returns None if no one challenges, otherwise returns the player who loses influence
        """
        if len(challenging_players) == 0:
            return None

        challenging_player = random.choice(challenging_players)

        # check if the action was a bluff
        if self.action_to_challenge.name == Action.BLOCK_ASSASSINATION or self.action_to_challenge.name == Action.BLOCK_STEAL_WITH_AMBASSADOR or self.action_to_challenge.name == Action.BLOCK_STEAL_WITH_CAPTAIN:  # type: ignore
            if (
                ACTION_TO_CARD[action_to_challenge.name]
                in self.players[self.blocking_agent].hidden_cards  # type: ignore
            ):
                # the action was not a bluff, so the challenger loses influence
                return challenging_player
            else:
                # the action was a bluff, so the challenged player loses influence
                return str(self.blocking_agent)
        else:
            if (
                ACTION_TO_CARD[action_to_challenge.name]
                in self.players[self.turn_agent].hidden_cards
            ):
                # the action was not a bluff, so the challenger loses influence
                return challenging_player
            else:
                # the action was a bluff, so the challenged player loses influence
                return str(self.turn_agent)  # type: ignore
