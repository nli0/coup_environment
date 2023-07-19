from typing import List, Tuple, Dict, Optional

from utils import ActionSpecification, AgentStatus, Cards


class Agent:
    def __init__(
        self,
        hidden_cards: List[Cards],
        coins: int = 2,
        name: Optional[str] = None,
        is_bot: bool = True,
        status: Optional[AgentStatus] = None,
    ) -> None:
        self.hidden_cards = hidden_cards
        self.coins = coins
        self.name = name
        self.is_bot = is_bot
        self.current_actions: List[ActionSpecification] = []
        self.status = status
        self.visible_cards = []
        self.observation: Optional[List[int]] = None

        self.ambassador_center_view: Optional[List[Cards]] = None
        self.history = ""

