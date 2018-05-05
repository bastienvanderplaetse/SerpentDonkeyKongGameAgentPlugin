from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey


class SerpentDonkeyKongGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        self.analytics_client = None

        # Genetic algorithms parameters
        self.population = 10
        self.evaluated_individuals = 0

    def setup_play(self):
        pass

    def handle_play(self, game_frame):
        if (self.game.api.not_running()):
        	print("not running")
        	self.game.api.run()
        else:
        	if (self.evaluated_individuals >= self.population):
        		self.visual_debugger.store_image_data(
                    game_frame.frame,
                    game_frame.frame.shape,
                    str(0)
                )
        		self.game.api.analyze_frame(game_frame)
        		if (self.evaluated_individuals % 3 == 0):
        			print("end generation")
        		elif (self.game.api.is_dead()):
        			print("Fake individual")
        			self.evaluated_individuals = self.evaluated_individuals + 1
        			self.game.api.replay()
        	else:
        		self.game.api.analyze_frame(game_frame)
        		if (self.game.api.is_in_menu()):
        			self.visual_debugger.store_image_data(
                    	game_frame.frame,
                    	game_frame.frame.shape,
                    	str(1)
                	)
        			frame, units = self.game.api.get_mario_frame(game_frame)
        			self.game.api.next()
        			if (frame != None):
        				print("DEMO")
        				self.input_controller.tap_key(KeyboardKey.KEY_SPACE)
        			self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
        			self.input_controller.tap_key(KeyboardKey.KEY_SPACE)
        			self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
        		elif (self.game.api.is_playing()):
        			self.visual_debugger.store_image_data(
                    	game_frame.frame,
                    	game_frame.frame.shape,
                    	str(2)
                	)
        			print("playing")
        		elif(self.game.api.is_dead()):
        			self.visual_debugger.store_image_data(
                    	game_frame.frame,
                    	game_frame.frame.shape,
                    	str(3)
                	)
        			print("Evaluate")
        			self.evaluated_individuals = self.evaluated_individuals + 1
        			self.game.api.replay()
