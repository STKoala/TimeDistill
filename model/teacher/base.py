class BaseTeacher:
    def forward(self, x):
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError

    def get_hidden_states(self, x):
        return None
