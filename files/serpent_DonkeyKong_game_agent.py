from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey

import sys

import numpy as np

from .helpers.multilayer_perceptron import UntrainedMLPClassifier


class SerpentDonkeyKongGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        self.analytics_client = None

        # Genetic algorithms parameters
        self.population = 4
        self.evaluated_individuals = 0

        # MLP
        self.mlp = UntrainedMLPClassifier()
        self.X = None;
        self.input_keys = {"LEFT": KeyboardKey.KEY_LEFT, "RIGHT": KeyboardKey.KEY_RIGHT, "UP": KeyboardKey.KEY_UP, "X": KeyboardKey.KEY_X}
        self.y = list(self.input_keys.keys())

        # Temp flags
        self.isInit = False

    def setup_play(self):
        pass

    def handle_play(self, game_frame):
        if (self.isInit):
            reduced_frame, units_array = self.game.api.get_mario_frame(game_frame)
            if (reduced_frame != None):
                nx, ny = units_array.shape
                units_array = units_array.reshape(nx * ny)
                res = self.mlp.predict([units_array])
                print(units_array)
                print(res)
        else :
            reduced_frame, units_array = self.game.api.get_mario_frame(game_frame)
            if (reduced_frame != None):
                nx, ny = units_array.shape
                self.X = np.zeros(nx*ny)

                coefs = []
                intercepts = []

                n_layers = np.random.randint(4)+2
                n_previous_neurons = nx * ny

                for i in range(0, n_layers-1):
                    n_neurons = np.random.randint(5)+1
                    coefs.append(np.random.rand(n_previous_neurons, n_neurons))
                    intercepts.append(np.random.rand(1, n_neurons))
                    n_previous_neurons = n_neurons

                coefs.append(np.random.rand(n_previous_neurons, 4))
                intercepts.append(np.random.rand(1, 4))

                self.mlp.prepare([self.X, self.X, self.X, self.X], self.y, coefs, intercepts)
                self.isInit = True

    def naivgation_handle_play(self, game_frame):
        if (self.game.api.not_running()):
            print("not running")
            self.game.api.run()
        else:
            if (self.evaluated_individuals >= self.population):
                self.game.api.analyze_frame(game_frame)
                if (self.evaluated_individuals % 3 == 0):
                    print("end generation")
                    sys.exit()
                elif (self.game.api.is_dead()):
                    print("Fake individual")
                    self.evaluated_individuals = self.evaluated_individuals + 1
                    self.game.api.replay()
            else:
                self.game.api.analyze_frame(game_frame)
                if (self.game.api.is_in_menu()):
                    frame, units = self.game.api.get_mario_frame(game_frame)
                    self.game.api.next()
                    if (frame != None):
                        print("DEMO")
                        self.input_controller.tap_key(KeyboardKey.KEY_SPACE)
                    self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
                    self.input_controller.tap_key(KeyboardKey.KEY_SPACE)
                    self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
                elif (self.game.api.is_playing()):
                    print("playing")
                elif(self.game.api.is_dead()):
                    print("Evaluate")
                    self.evaluated_individuals = self.evaluated_individuals + 1
                    self.game.api.replay()
