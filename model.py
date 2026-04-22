import torch
import torch.nn as nn

class ChessNet(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c,64,3,padding=1), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8,256), nn.ReLU()
        )
        self.policy = nn.Linear(256,4096)
        self.value = nn.Linear(256,1)

    def forward(self,x):
        x=self.conv(x)
        x=self.fc(x)
        return self.policy(x), torch.tanh(self.value(x))


def save_model(net, opt, replay, game_count, path):
    torch.save({
        "model": net.state_dict(),
        "opt": opt.state_dict(),
        "replay": list(replay),
        "game_count": game_count
    }, path)


def load_model(net, opt, replay, device, path):
    import os
    if not os.path.exists(path):
        print("no checkpoint found, starting fresh")
        return 0

    data = torch.load(path, map_location=device, weights_only=False)

    net.load_state_dict(data["model"])
    opt.load_state_dict(data["opt"])

    replay.clear()
    replay.extend(data["replay"])

    print("checkpoint loaded")

    return data.get("game_count", 0)