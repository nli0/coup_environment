from dataclasses import dataclass
from typing import Optional
from enum import Enum
from itertools import combinations


class Action(Enum):
    # Coup card turn actions
    AMBASSADOR = 0  # not fully specified
    ASSASSIN = 1  # not fully specified
    CAPTAIN = 2  # not fully specified
    DUKE = 3  # fully specified

    # Remaining turn actions
    INCOME = 4  # fully specified
    FOREIGN_AID = 5  # fully specified
    COUP = 6  # not fully specified

    # Reactionary actions
    CHALLENGE = 7  # fully specified
    NO_CHALLENGE = 8  # fully specified
    BLOCK_FOREIGN_AID = 9  # fully specified
    LET_FOREIGN_AID = 10  # fully specified
    BLOCK_STEAL_WITH_AMBASSADOR = 11  # fully specified
    BLOCK_STEAL_WITH_CAPTAIN = 12  # fully specified
    LET_STEAL = 13  # fully specified
    BLOCK_ASSASSINATION = 14  # fully specified
    LET_ASSASSINATION = 15  # fully specified

    # Reactions to reactions? idk
    LOSE_INFLUENCE = 16  # not fully specified

    # Ambassador
    AMBASSADOR_EXCHANGE_3 = 17  # not fully specified
    AMBASSADOR_EXCHANGE_4 = 18  # not fully specified


IS_FULLY_SPECIFIED = {
    Action.AMBASSADOR: True,
    Action.ASSASSIN: False,
    Action.CAPTAIN: False,
    Action.DUKE: True,
    Action.INCOME: True,
    Action.FOREIGN_AID: True,
    Action.COUP: False,
    Action.CHALLENGE: True,
    Action.NO_CHALLENGE: True,
    Action.BLOCK_FOREIGN_AID: True,
    Action.LET_FOREIGN_AID: True,
    Action.BLOCK_STEAL_WITH_AMBASSADOR: True,
    Action.BLOCK_STEAL_WITH_CAPTAIN: True,
    Action.LET_STEAL: True,
    Action.BLOCK_ASSASSINATION: True,
    Action.LET_ASSASSINATION: True,
    Action.LOSE_INFLUENCE: False,
    # TODO: ambassador stuff
}


# if not fully specified, do you specify the opponent or card?
SPECIFY_OPPONENT = {
    Action.AMBASSADOR: True,
    Action.ASSASSIN: True,
    Action.CAPTAIN: True,
    Action.COUP: True,
    Action.LOSE_INFLUENCE: False,
}


@dataclass
class ActionSpecification:
    """specifies an action for a coup player - the name of action, and the player it is targeting if not fully specified"""

    def __init__(self, name: Action, target: Optional[int] = None):
        self.name = name
        self.target = target  # if None, then the action is fully specified. target is between 0-5 for


class AgentStatus(Enum):
    TURN = 0
    STOLEN = 1
    ASSASSINATED = 2
    CAN_CHALLENGE = 3
    CAN_BLOCK_FOREIGN_AID = 4
    LOSE_INFLUENCE = 5
    AMBASSADOR_EXCHANGE = 6


class Cards(Enum):
    AMBASSADOR = 0
    ASSASSIN = 1
    CAPTAIN = 2
    CONTESSA = 3
    DUKE = 4


ACTION_TO_CARD = {
    Action.AMBASSADOR: Cards.AMBASSADOR,
    Action.ASSASSIN: Cards.ASSASSIN,
    Action.BLOCK_ASSASSINATION: Cards.CONTESSA,
    Action.BLOCK_FOREIGN_AID: Cards.DUKE,
    Action.DUKE: Cards.DUKE,
    Action.CAPTAIN: Cards.CAPTAIN,
    Action.BLOCK_STEAL_WITH_AMBASSADOR: Cards.AMBASSADOR,
    Action.BLOCK_STEAL_WITH_CAPTAIN: Cards.CAPTAIN,
}

CARD_TO_STRING = {
    Cards.AMBASSADOR: "Ambassador",
    Cards.ASSASSIN: "Assassin",
    Cards.CAPTAIN: "Captain",
    Cards.CONTESSA: "Contessa",
    Cards.DUKE: "Duke",
}


def map_combinations(n, r):
    """
    Creates a list of all combinations of size r of the first n integers.

    Args:
        n (int): The total number of integers.
        r (int): The size of combinations.

    Returns:
        list: A list of tuples representing the combinations.
    """
    numbers = range(n)
    combinations_list = list(combinations(numbers, r))
    return combinations_list
