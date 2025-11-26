class OODLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        with open(log_path, "w") as f:
            f.write("global_step,lambda,ood,episode\n")

    def log(self, global_step, lambda_val, ood, episode):
        with open(self.log_path, "a") as f:
            f.write(f"{global_step},{lambda_val},{ood},{episode}\n")

    def log_episode_end(self, episode):
        with open(self.log_path, "a") as f:
            f.write(f"EPISODE_END,{episode}\n")