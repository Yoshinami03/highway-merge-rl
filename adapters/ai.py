class AIAdapter:
    def __init__(self, policy):
        self.policy = policy
    def act(self, obs):
        a, _ = self.policy.predict(obs, deterministic=False)
        return a
