import time
import random
from torch.utils.tensorboard import SummaryWriter

# Define the number of epochs and iterations per epoch
num_epochs = 5
iterations_per_epoch = 1000

# Initialize the TensorBoard writer
writer = SummaryWriter()

# Fake training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    print(f"Epoch {epoch+1}/{num_epochs}")
    print('-' * 10)

    for iteration in range(iterations_per_epoch):
        # Simulating some training by waiting a bit
        time.sleep(0.1)
        
        # Creating fake loss and accuracy values
        loss = random.uniform(0.1, 2.0)  # Fake loss value
        accuracy = random.uniform(70, 100)  # Fake accuracy value
        
        epoch_loss += loss

        # Log loss and accuracy to TensorBoard
        writer.add_scalar('Loss/train', loss, epoch * iterations_per_epoch + iteration)
        writer.add_scalar('Accuracy/train', accuracy, epoch * iterations_per_epoch + iteration)

        # Print the fake iteration stats
        print(f"Iteration {iteration+1}/{iterations_per_epoch} - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Calculate average loss for the epoch
    epoch_loss /= iterations_per_epoch
    print(f"\nSummary for Epoch {epoch+1}: Average Loss: {epoch_loss:.4f}\n")

    # Log average loss for the epoch to TensorBoard
    writer.add_scalar('Loss/epoch', epoch_loss, epoch)

print("Training completed.")

# Close the TensorBoard writer
writer.close()