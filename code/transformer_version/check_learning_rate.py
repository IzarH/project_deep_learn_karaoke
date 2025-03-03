import matplotlib.pyplot as plt
import torch
from torch.optim import Adam


# Recreate your WarmupLinearSchedule class
class WarmupLinearSchedule(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, self.get_lr_lambda)

    def get_lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.total_steps - step) / float(max(1, self.total_steps - self.warmup_steps)))

# Parameters for the scheduler
initial_lr = 0.001  # Default learning rate
epochs = 1000       # Adjust based on your training setup
batch_size = 64
steps_per_epoch = 100  # Set this to an approximate or actual step count for each epoch
total_steps = epochs * steps_per_epoch
warmup_steps = int(0.05 * total_steps)

# Dummy optimizer to test the scheduler
model_params = [torch.zeros(1)]  # Replace with model params in real usage
optimizer = Adam(model_params, lr=initial_lr)
scheduler = WarmupLinearSchedule(optimizer, warmup_steps, total_steps)

# Plot the learning rate
learning_rates = []
for step in range(total_steps):
    optimizer.step()  # Dummy step to bypass the warning
    scheduler.step()
    learning_rates.append(scheduler.get_last_lr()[0])

plt.figure(figsize=(10, 6))
plt.plot(range(total_steps), learning_rates, label="Learning Rate")
plt.xlabel("Training Steps")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule with Warmup")
plt.legend()
plt.savefig("learning_rate_schedule.png")