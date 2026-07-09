import math
import random

from utils.communication import encode_message, decode_message


class WeightedVoterModel:

    def __init__(self, num_options=4):
        self.num_options = num_options

        self.opinion = -1
        self.q_est = 0.0

        self.msgs_rx_this_tick = 0
        self.msgs_rx_total = 0
        self.msgs_tx_total = 0

    async def perform_weighted_voter_model(self, robot):
        self.msgs_rx_this_tick = 0

        if self.opinion >= 0:
            await robot.send(encode_message(self.opinion, self.q_est))
            self.msgs_tx_total += 1

        message = await robot.receive()

        if message is not None:
            self.msgs_rx_this_tick += 1
            self.msgs_rx_total += 1

            opinion, quality = decode_message(message)
            self.process_received_opinion(opinion, quality)

        return self.opinion
    