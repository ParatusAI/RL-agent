"""
rl_agent_api_v3.py  –  Flow-Temp Controller with FWHM Optimisation
────────────────────────────────────────────────────────────────
Action (3)      : [csbr_flow, pbbr2_flow, temperature]
Observation (7) : [cur_PLQY, cur_lambda, cur_FWHM,
                   cur_temperature,
                   tgt_PLQY, tgt_lambda, tgt_FWHM]
"""

# ─────────────── imports ───────────────
import threading
from pathlib import Path
import numpy as np
import gymnasium as gym
import pandas as pd

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# ─────────────── 1. Custom Gym environment ───────────────


class FlowTempFwhmEnv(gym.Env):
    """Controls CsBr, PbBr₂ flow & temperature to hit PLQY, λ, FWHM targets."""
    ACT_LOW = np.array([0.10, 0.10,  60.], dtype=np.float32)
    ACT_HIGH = np.array([0.35, 0.35, 120.])
    OBS_LOW = np.array([0., 510., 10., 60.,   0., 510., 10.], dtype=np.float32)
    OBS_HIGH = np.array([100., 530., 40., 120., 100., 530., 40.])

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(self.ACT_LOW, self.ACT_HIGH)
        self.observation_space = gym.spaces.Box(self.OBS_LOW, self.OBS_HIGH)
        self.max_steps = 40
        self.state, self.last_action, self.t = None, None, 0

    # synthetic chemistry response
    @staticmethod
    def _react(action):
        cs, pb, T = action
        plqy = np.clip(25 + 450*(cs-pb) - 0.4*abs(T-90) +
                       np.random.randn()*2, 0, 100)
        lam = np.clip(540 - 65*(cs+pb) + 0.25*(T-90) +
                      np.random.randn()*0.6, 510, 530)
        fwhm = np.clip(35 - 40*abs(cs-pb) + 0.1*abs(T-90) +
                       np.random.randn()*0.4, 10, 40)
        return float(plqy), float(lam), float(fwhm)

    # Gymnasium ≥0.29 reset signature
    def reset(self, *, seed=None, options=None,
              tgt_plqy=None, tgt_lambda=None, tgt_fwhm=None):
        super().reset(seed=seed)
        rng = self.np_random

        tgt_plqy = float(rng.uniform(70, 90)) if tgt_plqy is None else tgt_plqy
        tgt_lambda = float(rng.uniform(515, 525)
                           ) if tgt_lambda is None else tgt_lambda
        tgt_fwhm = float(rng.uniform(15, 25)) if tgt_fwhm is None else tgt_fwhm

        cur_plqy = float(rng.uniform(25, 45))
        cur_lambda = float(rng.uniform(515, 525))
        cur_fwhm = float(rng.uniform(20, 30))
        cur_T = float(rng.uniform(80, 100))

        self.state = np.array([cur_plqy, cur_lambda, cur_fwhm, cur_T,
                               tgt_plqy, tgt_lambda, tgt_fwhm], dtype=np.float32)
        self.last_action, self.t = None, 0
        return self.state, {}              # (obs, info) tuple

    def step(self, action):
        self.t += 1
        action = np.clip(action, self.ACT_LOW, self.ACT_HIGH)

        cur_plqy, cur_lam, cur_fwhm = self._react(action)
        cur_T = action[2]
        tgt_plqy, tgt_lambda, tgt_fwhm = self.state[4:]

        obs_next = np.array([cur_plqy, cur_lam, cur_fwhm, cur_T,
                             tgt_plqy, tgt_lambda, tgt_fwhm], dtype=np.float32)

        w1, w2, w3, w4 = 1.0, 0.2, 0.2, 0.002
        reward = -(w1*(cur_plqy-tgt_plqy)**2
                   + w2*(cur_lam - tgt_lambda)**2
                   + w3*(cur_fwhm-tgt_fwhm)**2
                   + w4*abs(cur_T-90))
        if self.last_action is not None:
            reward -= 0.01*np.linalg.norm(action-self.last_action)

        self.state, self.last_action = obs_next, action
        terminated = False
        truncated = self.t >= self.max_steps
        return obs_next, reward, terminated, truncated, {}


# ─────────────── 2. Build & check env, train or load PPO ───────────────
raw_env = FlowTempFwhmEnv()
check_env(raw_env, warn=True)           # passes new Gym API spec

ENV = raw_env                            # SB3 ≥2.3 uses new API, so no wrapper

MODELDIR = Path("models")
MODELDIR.mkdir(exist_ok=True)
MODEL_PATH = MODELDIR / "ppo_flow_temp_fwhm.zip"


def load_or_train():
    if MODEL_PATH.exists():
        return PPO.load(MODEL_PATH, env=ENV)

    model = PPO("MlpPolicy", ENV,
                learning_rate=3e-4, n_steps=2048,
                batch_size=256, verbose=1)   # tensorboard_log removed
    model.learn(50_000)                      # lower steps for quick start
    model.save(MODEL_PATH)
    return model


MODEL = load_or_train()
LOCK = threading.Lock()

# ─────────────── 3. FastAPI layer ───────────────
app = FastAPI(title="RL Flow-Temp-FWHM Controller")


class ResetBody(BaseModel):
    target_plqy: float
    target_lambda: float
    target_fwhm: float


class StepBody(BaseModel):
    plqy: float
    emission: float
    fwhm: float
    temperature: float


class TrainBody(BaseModel):
    n_steps: int = 10000


@app.get("/status")
def status():
    return {"ok": True,
            "algo": "PPO",
            "action_low": ENV.ACT_LOW.tolist(),
            "action_high": ENV.ACT_HIGH.tolist()}


@app.post("/reset")
def api_reset(body: ResetBody):
    with LOCK:
        obs, _ = ENV.reset(tgt_plqy=body.target_plqy,
                           tgt_lambda=body.target_lambda,
                           tgt_fwhm=body.target_fwhm)
    return {"observation": obs.tolist()}


@app.post("/step")
def api_step(body: StepBody):
    with LOCK:
        ENV.state[:4] = [body.plqy, body.emission,
                         body.fwhm, body.temperature]
        action, _ = MODEL.predict(ENV.state, deterministic=False)
        next_obs, reward, terminated, truncated, _ = ENV.step(action)
    return {"action": {"csbr": float(action[0]),
                       "pbbr2": float(action[1]),
                       "temperature": float(action[2])},
            "reward": float(reward),
            "done":   terminated or truncated,
            "next_obs": next_obs.tolist()}


@app.post("/train")
def api_train(body: TrainBody):
    def _bg():
        with LOCK:
            MODEL.set_env(ENV)
            MODEL.learn(body.n_steps, reset_num_timesteps=False)
            MODEL.save(MODEL_PATH)
    threading.Thread(target=_bg, daemon=True).start()
    return {"msg": f"training launched for {body.n_steps} steps"}


@app.post("/predict-from-csv")
async def predict_from_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    responses = []

    for _, row in df.iterrows():
        state_input = [row["plqy"], row["emission"],
                       row["fwhm"], row["temperature"]]
        with LOCK:
            ENV.state[:4] = state_input
            action, _ = MODEL.predict(ENV.state, deterministic=False)
        responses.append({
            "input": {
                "plqy": row["plqy"],
                "emission": row["emission"],
                "fwhm": row["fwhm"],
                "temperature": row["temperature"]
            },
            "predicted_action": {
                "csbr_flow": float(action[0]),
                "pbbr2_flow": float(action[1]),
                "temperature": float(action[2])
            }
        })

    return {"predictions": responses}


# ─────────────── 4. Run server ───────────────
if __name__ == "__main__":
    uvicorn.run("rl_agent_api:app", host="0.0.0.0", port=8000)
