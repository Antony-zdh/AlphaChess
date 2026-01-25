import os
import pygame
import chess

class ChessGUI:
    """ Class to handle the graphical user interface """
    def __init__(self, board: chess.Board):
        self.board = board
        self.square_size = 80 # The size of one square on the chessboard
        self.height = self.square_size * 8
        self.width = self.square_size * 8
        self.selected_square = None
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Alpha Chess Game')
        self.images = self._load_images()

    
    def _load_images(self):
        """ Load images for the chess pieces. """
        pieces = ['bd', 'bl', 'kd', 'kl', 'nd', 'nl',
                  'pd', 'pl', 'qd', 'ql', 'rd', 'rl']
        images = {}
        for piece in pieces:
            assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
            images[piece] = pygame.transform.scale(
                surface=pygame.image.load(os.path.join(assets_dir, f'Chess_{piece}t60.png')),
                size=(self.square_size, self.square_size)
            )
        return images


    def draw_board(self):
        """ Draw the chess board and pieces. """
        colors = [pygame.Color(235, 209, 166), pygame.Color(165, 117, 81)]
        for r in range(8):
            for c in range(8):
                color = colors[((r + c) % 2)]
                pygame.draw.rect(self.screen, color,
                                 pygame.Rect(c*self.square_size, r*self.square_size,
                                             self.square_size, self.square_size))
        self.draw_pieces()
        pygame.display.flip()
    
    
    def draw_pieces(self):
        """ Draw the chess pieces on the board. """
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                piece_color = 'd' if piece.color == chess.BLACK else 'l'
                piece_type = piece.symbol().lower()
                piece_key = piece_type + piece_color
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                self.screen.blit(self.images[piece_key],
                                 pygame.Rect(col*self.square_size, row*self.square_size,
                                             self.square_size, self.square_size))


    def locate_click(self, pos):
            """ Locate mouse click events to make moves. """
            col = pos[0] // self.square_size
            row = 7 - (pos[1] // self.square_size)
            square = chess.square(col, row)
            return square
    

    def get_human_move(self, color):
        """ Handle user clicks. """
        if color != self.board.turn:
            return None # Ignore events if it's not the current player's turn
        while True:
            # We must redraw/update to keep the window from freezing "Not Responding"
            self.draw_board()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    square = self.locate_click(pos)
                    # To move a piece, first select the piece
                    # then select the destination square
                    if self.selected_square is None:
                        self.selected_square = square
                    else:
                        # Translate the move to uci format
                        move = chess.Move(self.selected_square, square)
                        self.selected_square = None
                        if move in self.board.legal_moves:
                            return move

    
    def wait_for_quit(self):
        """ Wait for the user to close the window. """
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
        pygame.quit()

    
    def display_countdown(self, role):
        """ Display a countdown timer when a player is thinking. """
        font = pygame.font.SysFont(None, 48)
        # TODO: Figure out a non-blocking way to do this
        countdown_time = 3  # seconds
        for i in range(countdown_time, 0, -1):
            self.draw_board()
            text = font.render(f"{role} is thinking... {i}", True, (255, 0, 0))
            self.screen.blit(text, (10, 10))
            pygame.display.flip()
            pygame.time.delay(1000)