"""Resumable, deterministic ID-supervised training on explicit sequence splits."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from berrytrack.data.mot import MOTSequence
from berrytrack.embeddings.model import CompactEncoder, supervised_contrastive_loss


class IdentityCrops(Dataset):
    def __init__(self, sequences: list[Path], max_samples: int = 2048) -> None:
        self.samples: list[tuple[Path, np.ndarray, int]] = []
        identity = 0
        for root in sequences:
            sequence = MOTSequence.load(root)
            gt = sequence.gt
            mapping = {int(i): identity + k for k, i in enumerate(np.unique(gt[:, 1]))}
            identity += len(mapping)
            for row in gt:
                if row[6] > 0:
                    self.samples.append(
                        (sequence.image_path(int(row[0])), row[2:6], mapping[int(row[1])])
                    )
        # Preserve repeated identities for contrastive pairs; deterministic temporal sampling.
        if len(self.samples) > max_samples:
            indices = np.linspace(0, len(self.samples) - 1, max_samples, dtype=int)
            self.samples = [self.samples[i] for i in indices]
        if not self.samples:
            raise ValueError("Training requires real annotated identity crops")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, box, label = self.samples[index]
        x, y, w, h = box
        with Image.open(path) as image:
            crop = image.convert("RGB").crop((x - 1, y - 1, x - 1 + w, y - 1 + h)).resize((32, 32))
            arr = np.asarray(crop, dtype=np.float32).copy() / 255
        return torch.from_numpy(arr).permute(2, 0, 1), label


def train_encoder(
    sequences: list[Path],
    output: Path,
    dimension: int = 64,
    steps: int = 30,
    device: str = "cpu",
    seed: int = 42,
    resume: bool = True,
    max_seconds: int = 120,
) -> dict:
    if steps < 1 or max_seconds < 1:
        raise ValueError("Training limits must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True, warn_only=True)
    data = IdentityCrops(sequences)
    # Each image's crop has a second augmented view, guaranteeing a positive pair.
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(data, batch_size=min(32, len(data)), shuffle=True, generator=generator)
    model = CompactEncoder(dimension).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "encoder.pt"
    start_step = 0
    loss_history: list[float] = []
    if resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state["dimension"] != dimension or state["sequences"] != [str(p) for p in sequences]:
            raise ValueError("Resume checkpoint uses different dimensions or data")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        loss_history = state["loss_history"]
        generator.set_state(state["loader_rng"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
    started = time.perf_counter()
    step = start_step
    iterator = iter(loader)
    while step < start_step + steps and time.perf_counter() - started < max_seconds:
        try:
            crops, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            crops, labels = next(iterator)
        crops, labels = crops.to(device), labels.to(device)
        view = (crops.flip(-1) * 0.95 + 0.025).clamp(0, 1)
        features = model(torch.cat((crops, view)))
        loss = supervised_contrastive_loss(features, labels.repeat(2))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach()))
        step += 1
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "dimension": dimension,
            "seed": seed,
            "sequences": [str(p) for p in sequences],
            "loss_history": loss_history,
            "loader_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
        }
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(checkpoint)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    result = {
        "step": step,
        "resumed_from": start_step,
        "loss": loss_history[-1] if loss_history else None,
        "loss_history": loss_history,
        "dimension": dimension,
        "seed": seed,
        "sequences": [str(p) for p in sequences],
        "samples": len(data),
        "device": device,
        "parameters": sum(p.numel() for p in model.parameters()),
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()
        if device.startswith("cuda")
        else None,
        "protocol": "development training smoke; loss is not a tracking accuracy metric",
        "resume_semantics": "weights, optimizer, step and RNG restored; data iterator restarts",
    }
    (output / "training.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


class TorchAppearance:
    def __init__(self, checkpoint: str | Path, device: str = "cpu") -> None:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        self.model = CompactEncoder(state["dimension"]).to(device).eval()
        self.model.load_state_dict(state["model"])
        self.device = device
        self.dimension = state["dimension"]

    def extract(self, frame, boxes):
        import cv2

        crops = []
        h, w = frame.shape[:2]
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            crop = frame[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
            if not crop.size:
                crop = np.zeros((32, 32, 3), np.uint8)
            crops.append(cv2.cvtColor(cv2.resize(crop, (32, 32)), cv2.COLOR_BGR2RGB))
        if not crops:
            return np.empty((0, self.dimension), np.float32)
        tensor = (
            torch.as_tensor(np.stack(crops), device=self.device).permute(0, 3, 1, 2).float() / 255
        )
        with torch.inference_mode():
            return self.model(tensor).cpu().numpy()
