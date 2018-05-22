from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey

import sys

import numpy as np

from .helpers.neuro_evolution import NEAT


class SerpentDonkeyKongGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        self.analytics_client = None

        # Genetic algorithms parameters
        self.evaluated_individuals = 0
        self.neat = NEAT(5)
        self.old_keys = [0,0,0,0,0]
        self.input_keys = {0: KeyboardKey.KEY_LEFT, 1: KeyboardKey.KEY_RIGHT, 2: KeyboardKey.KEY_UP, 3: KeyboardKey.KEY_DOWN, 4: KeyboardKey.KEY_X}

        self.counter = 0
        self.old_posX = 0

    def setup_play(self):
        pass

    def handle_play(self, game_frame):
        if (self.game.api.not_running()):
            self.game.api.run()
        else :
            locations = self.game.api.analyze_frame(game_frame)
            if (self.neat.generation_finished()):
                if (self.evaluated_individuals %3 == 0):
                    self.neat.next_generation()
                elif (self.game.api.is_dead()):
                    self.evaluated_individuals = self.evaluated_individuals + 1
                    self.game.api.replay()
            else :
                if (self.game.api.is_in_menu()):
                    self.input_controller.tap_key(KeyboardKey.KEY_R)
                    self.game.api.next()
                    self.neat.prepare_next()
                    if (locations[0] != None):
                        self.input_controller.tap_key(KeyboardKey.KEY_SPACE)
                    self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
                    self.input_controller.tap_key(KeyboardKey.KEY_SPACE)
                    self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
                    self.old_posX = 0
                    self.counter = 0
                elif (self.game.api.is_playing()):
                    reduced_frame, projection_matrix = self.game.api.get_projection_matrix(game_frame, locations[0])
                    keys = self.neat.feed(projection_matrix)
                    self._press_keys(keys)
                    self.old_keys = keys
                    if(keys == [0,0,0,0,0] or (locations[0] != None and projection_matrix[len(projection_matrix)-1] != 3 and  abs(self.old_posX-locations[0][1]) <= 5)):
                        self.counter = self.counter + 1
                        if(self.counter == 7):
                            self.game.api.win()
                            keys = [0,0,0,0,0]
                            self._press_keys(keys)
                            self.old_keys = keys
                            self.input_controller.tap_key(KeyboardKey.KEY_R, duration=0.5)
                            self.game.api.run()
                            score = self.game.api.get_final_position(locations[0])
                            self.neat.fitness(score)
                            self.evaluated_individuals = 0
                    else :
                        self.counter = 0
                        if (locations[0] != None):
                            self.old_posX = locations[0][1]
                elif (self.game.api.is_dead()):
                    position_dead = self.game.api.get_position_dead(game_frame)
                    self.neat.fitness(position_dead)
                    self.evaluated_individuals = self.evaluated_individuals + 1
                    self.game.api.replay()
                elif (self.game.api.has_won()):
                    keys = [0,0,0,0,0]
                    self._press_keys(keys)
                    self.old_keys = keys
                    self.input_controller.tap_key(KeyboardKey.KEY_R, duration=0.5)
                    self.game.api.run()
                    self.neat.fitness([0,6000])
                    self.evaluated_individuals = 0

    def _press_keys(self, keys):
        for i in range(len(keys)-1):
            if (keys[i] == 1 and self.old_keys[i] == 0):
                self.input_controller.press_key(self.input_keys[i])
            elif (keys[i] == 0 and self.old_keys[i] == 1) :
                self.input_controller.release_key(self.input_keys[i])
        if (keys[len(keys)-1] == 1):
            self.input_controller.tap_key(self.input_keys[len(keys)-1])