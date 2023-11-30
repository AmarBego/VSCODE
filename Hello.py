# TICTOE

board = [' ' for _ in range(9)]

def print_board():
    for i in range(0, 9, 3):
        print(board[i], board[i+1], board[i+2])

def is_game_over():
    for i in range(0, 9, 3):
        if board[i] == board[i+1] == board[i+2] != ' ':
            return True
    for i in range(3):
        if board[i] == board[i+3] == board[i+6] != ' ':
            return True
    if board[0] == board[4] == board[8] != ' ' or board[2] == board[4] == board[6] != ' ':
        return True
    if ' ' not in board:
        return True
    return False

def make_move():
    with open('history.txt', 'r') as f:
        history = f.read().splitlines()

    current_board = ''.join(board)

    for line in history:
        past_board, move, result = line.split(',')
        if past_board == current_board and result == 'win':
            board[int(move)] = 'O'
            return

    for i in range(9):
        if board[i] == ' ':
            board[i] = 'O'
            break

while not is_game_over():
    move = int(input("Enter your move (0-8): "))
    if board[move] != ' ':
        print("Invalid move!")
        continue
    board[move] = 'X'
    if is_game_over():
        break
    make_move()
    print_board()
print("Game over!")