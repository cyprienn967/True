from typing import List, Dict

class ConversationStore:
 
    def __init__(self):
        self.state: Dict[str, List[str]] = {}

    def init_conversation(self, conversation_id: str):
        if conversation_id not in self.state:
            self.state[conversation_id] = []

    def add_step(self, conversation_id: str, step_text: str):
        self.init_conversation(conversation_id)
        self.state[conversation_id].append(step_text)

    def get_steps(self, conversation_id: str) -> List[str]:
        self.init_conversation(conversation_id)
        return self.state[conversation_id]

conversation_store = ConversationStore()
