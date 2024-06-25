class AlphaBetaAgent:
    def __init__(self, token):
        self.my_token = token
        if self.my_token == 'x':
            self.opponent_token = 'o'
        else:
            self.opponent_token = 'x'

    def decide(self, connect4):
        possible_moves = connect4.possible_drops()
        best_move = None
        alpha = float('-inf')
        beta = float('inf')

        for move in possible_moves:
            # Make a hypothetical move
            connect4.drop_token(move)

            # Calculate the score for this move using Alfa-Beta Pruning
            score = self.alphabeta(connect4, depth=3, alpha=alpha, beta=beta, maximizing_player=False)

            # Undo the hypothetical move
            connect4.undo_last_drop(move)

            # Update best move if needed
            if score > alpha:
                alpha = score
                best_move = move

        return best_move

    def alphabeta(self, connect4, depth, alpha, beta, maximizing_player):
        if depth == 0 or connect4.game_over:
            # Calculate the heuristic score here (evaluation function)
            return self.evaluate(connect4)

        if maximizing_player:
            max_eval = float('-inf')
            for move in connect4.possible_drops():
                connect4.drop_token(move)
                eval = self.alphabeta(connect4, depth - 1, alpha, beta, False)
                connect4.undo_last_drop(move)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in connect4.possible_drops():
                connect4.drop_token(move)
                eval = self.alphabeta(connect4, depth - 1, alpha, beta, True)
                connect4.undo_last_drop(move)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate(self, connect4):
        if connect4.game_over:
            if connect4.wins == self.my_token:
                return 1
            elif connect4.wins == self.opponent_token:
                return -1
            else:
                return 0

        # Inicjalizacja wartości oceny
        score = 0

        # Ocena na podstawie bliskości do wygranej
        for four in connect4.iter_fours():
            if four.count(self.my_token) == 3 and four.count('_') == 1:
                score += 100  # Blisko wygranej
            elif four.count(self.opponent_token) == 3 and four.count('_') == 1:
                score -= 100  # Blisko przegranej

        # Ocena na podstawie zajęcia środkowych kolumn
        center_col = connect4.center_column()
        my_center_tokens = center_col.count(self.my_token)
        opponent_center_tokens = center_col.count(self.opponent_token)
        score += (my_center_tokens - opponent_center_tokens) * 10

        # Znormalizowanie wyniku do przedziału od -1 do 1
        max_possible_score = 100 * 4 + 10 * 3
        min_possible_score = -100 * 4 - 10 * 3
        normalized_score = (score - min_possible_score) / (max_possible_score - min_possible_score) * 2 - 1

        if normalized_score > 1:
            return 0.999
        elif normalized_score < -1:
            return -0.999

        return normalized_score
