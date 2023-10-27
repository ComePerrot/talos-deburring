class RiccatiController:
    def __init__(self, state,torque, xref, riccati):
        self.state = state
        self.update_references(torque, xref, riccati)

    def update_references(self, torque, xref, riccati):
        self.torque_ff = torque
        self.x0 = xref
        self.riccati_gain = riccati

    def step(self, x_measured):
        return self.torque_ff + self.riccati_gain @ self.state.diff(x_measured, self.x0)
