from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

root_path = Path(".")
data_path = root_path / "Data"
session_time = datetime.now().strftime("%Y%m%d%H%M")
results_folder = data_path.joinpath(f"results/{session_time}")
results_folder.mkdir(exist_ok=True, parents=True)
summary_folder = data_path.joinpath(f"summaries/{session_time}")
summary_folder.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(str(summary_folder))
