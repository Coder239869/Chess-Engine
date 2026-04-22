import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import random
import math
import os
from collections import deque

from model import ChessNet, save_model, load_model

# =========================
# FLAGS
# =========================
USE_NOISE = False
USE_NEGAMAX = False
CURRICULUM = True

GAME_COUNT = 0 
CURRICULUM_PHASE1 = 2
CURRICULUM_PHASE2 = 800

CHECKPOINT = "checkpoint.pt"
TRAIN_PGN = "training_games.pgn"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# MENU HELPERS
# =========================
def print_menu():
    print("\n" + "="*40)
    print("♟️  CHESS ENGINE CONTROL PANEL")
    print("="*40)
    print(f"1. Train (workers)")
    print(f"2. Play vs Engine")
    print(f"3. Toggle Noise      -> {USE_NOISE}")
    print(f"4. Toggle Negamax    -> {USE_NEGAMAX}")
    print(f"5. Toggle Curriculum -> {CURRICULUM}")
    print(f"6. Save checkpoint")
    print(f"7. Exit")
    print("="*40)

# =========================
# TEMPERATURE SCHEDULE
# =========================
def temperature(move_count):
    if move_count < 10:
        return 1.5
    elif move_count < 30:
        return 1.0
    else:
        return 0.2

# =========================
# ENCODE
# =========================
def encode(board):
    planes=[]
    for color in [chess.WHITE,chess.BLACK]:
        for pt in [chess.PAWN,chess.KNIGHT,chess.BISHOP,
                   chess.ROOK,chess.QUEEN,chess.KING]:
            p=torch.zeros((8,8))
            for sq in board.pieces(pt,color):
                r,f=divmod(sq,8)
                p[r][f]=1
            planes.append(p)
    planes.append(torch.ones((8,8)) if board.turn else torch.zeros((8,8)))
    return torch.stack(planes)

INPUT_CHANNELS = len(encode(chess.Board()))

# =========================
# MODEL
# =========================
net = ChessNet(INPUT_CHANNELS).to(device)
opt = torch.optim.Adam(net.parameters(), lr=0.001)

# =========================
# CHECKPOINT
# =========================
def save_checkpoint(replay):
    torch.save({
        "model": net.state_dict(),
        "opt": opt.state_dict(),
        "replay": list(replay),
        "game_count": GAME_COUNT
    }, CHECKPOINT)

def load_checkpoint(replay):
    global GAME_COUNT

    if not os.path.exists(CHECKPOINT):
        print("no checkpoint found")
        return

    data = torch.load(CHECKPOINT, map_location=device, weights_only=False)

    net.load_state_dict(data["model"])
    opt.load_state_dict(data["opt"])

    replay.clear()
    replay.extend(data["replay"])

    GAME_COUNT = data.get("game_count", 0)

    print("checkpoint loaded")

# =========================
# NEGAMAX
# =========================
VAL = {
    chess.PAWN:100,
    chess.KNIGHT:300,
    chess.BISHOP:300,
    chess.ROOK:500,
    chess.QUEEN:900,
    chess.KING:0
}

def eval_simple(board):
    s=0
    for p,v in VAL.items():
        s+=len(board.pieces(p,chess.WHITE))*v
        s-=len(board.pieces(p,chess.BLACK))*v
    return s if board.turn else -s

def negamax(board,depth,alpha,beta):
    if depth==0 or board.is_game_over():
        return eval_simple(board)

    best=-1e9
    for m in board.legal_moves:
        board.push(m)
        val=-negamax(board,depth-1,-beta,-alpha)
        board.pop()
        best=max(best,val)
        alpha=max(alpha,val)
        if alpha>=beta:
            break
    return best

def negamax_move(board):
    best=None
    bestv=-1e9
    for m in board.legal_moves:
        board.push(m)
        v=-negamax(board,2,-1e9,1e9)
        board.pop()
        if v>bestv:
            bestv=v
            best=m
    return best

# =========================
# CURRICULUM
# =========================
def mode():
    if not CURRICULUM:
        return "mcts" if not USE_NEGAMAX else "negamax"

    if GAME_COUNT < CURRICULUM_PHASE1:
        return "negamax"
    elif GAME_COUNT < CURRICULUM_PHASE2:
        return "hybrid"
    return "mcts"

# =========================
# MCTS NODE
# =========================
class Node:
    def __init__(self,board,parent=None,prior=0):
        self.board=board
        self.parent=parent
        self.prior=prior
        self.children={}
        self.visits=0
        self.value=0

# =========================
# MCTS (with diversity fixes)
# =========================
def expand(node):
    # fake NN stub (keep your real one)
    p = torch.rand(4096)
    v = random.uniform(-1,1)

    legal = list(node.board.legal_moves)

    # ROOT NOISE (IMPORTANT FIX)
    if USE_NOISE:
        alpha = 0.3
        eps = 0.25
        noise = [random.gammavariate(alpha,1) for _ in legal]
        s = sum(noise)+1e-8

    for i,m in enumerate(legal):
        b=node.board.copy()
        b.push(m)

        idx=m.from_square*64+m.to_square
        prior=float(p[idx])

        if USE_NOISE:
            prior = (1-eps)*prior + eps*(noise[i]/s)

        node.children[m]=Node(b,node,prior)

    return v

def select(node):
    best=None
    best_score=-1e9
    for m,ch in node.children.items():
        u=1.4*ch.prior*math.sqrt(node.visits+1)/(1+ch.visits)
        q=ch.value/(1+ch.visits)
        s=u+q
        if s>best_score:
            best_score=s
            best=(m,ch)
    return best

def backprop(node,v):
    while node:
        node.visits+=1
        node.value+=v
        v=-v
        node=node.parent

def mcts(board):
    root=Node(board)

    # opening randomness (BIG FIX)
    if USE_NOISE:
        for _ in range(random.randint(0,4)):
            if board.is_game_over():
                break
            board.push(random.choice(list(board.legal_moves)))

    expand(root)

    for _ in range(20):
        node=root
        while node.children:
            m,node=select(node)

        if not node.board.is_game_over():
            v=expand(node)
        else:
            v=0

        backprop(node,v)

    return root

def pick(root,temp=1.0):
    moves=list(root.children.keys())
    visits=torch.tensor([root.children[m].visits for m in moves],dtype=torch.float32)

    if temp==0:
        return moves[int(torch.argmax(visits))]

    probs=visits**(1/temp)
    probs/=probs.sum()
    return random.choices(moves,probs.tolist())[0]

# =========================
# SELF PLAY
# =========================
def self_play(replay):
    global GAME_COUNT

    board=chess.Board()
    game=chess.pgn.Game()
    node=game

    for _ in range(200):
        if board.is_game_over():
            break

        m=mode()

        if m=="negamax":
            move=negamax_move(board)
        else:
            root=mcts(board)
            move=pick(root, temperature(len(board.move_stack)) if USE_NOISE else 0)

        board.push(move)
        node=node.add_variation(move)

    result=board.result()

    reward = 1 if result=="1-0" else -1 if result=="0-1" else 0
    reward += random.uniform(-0.05,0.05)  # collapse fix

    replay.append((board.fen(),reward))

    with open(TRAIN_PGN,"a") as f:
        f.write(str(game)+"\n\n")

    GAME_COUNT+=1

# =========================
# WORKER
# =========================
def worker(i,replay):
    while True:
        self_play(replay)

# =========================
# TRAIN LOOP
# =========================
def train_loop(replay):
    procs=[]

    for i in range(5):
        p=mp.Process(target=worker,args=(i,replay))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

# =========================
# PLAY
# =========================
def play(replay):
    board=chess.Board()

    while not board.is_game_over():
        print(board,"\n")

        move=input("SAN: ")
        try:
            board.push(board.parse_san(move))
        except:
            continue

        if USE_NEGAMAX:
            engine=negamax_move(board)
        else:
            root=mcts(board)
            engine=pick(root)

        print("engine:",board.san(engine))
        board.push(engine)

    print("Game:",board.result())

# =========================
# MAIN
# =========================
def main():
    global USE_NOISE, USE_NEGAMAX, CURRICULUM

    replay=[]

    load_checkpoint(replay)

    while True:
        print_menu()
        c=input(">> ")

        if c=="1":
            train_loop(replay)

        elif c=="2":
            play(replay)

        elif c=="3":
            USE_NOISE=not USE_NOISE

        elif c=="4":
            USE_NEGAMAX=not USE_NEGAMAX

        elif c=="5":
            CURRICULUM=not CURRICULUM

        elif c=="6":
            save_checkpoint(replay)
            print("saved")

        elif c=="7":
            break

if __name__=="__main__":
    mp.set_start_method("spawn")
    main()